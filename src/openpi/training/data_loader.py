from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from torch.utils._pytree import tree_map

import openpi.models.model as _model
from openpi.training.collector_frames import LegacyCollectorFrames
from openpi.training.collector_frames import sequence_keys
import openpi.training.config as _config
from openpi.training.franka_multirate import MultirateFrankaDataset
from openpi.training.lerobot_utils import resolve_dataset_root
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class SubsetDataset(Dataset[T_co]):
    """View of ``dataset`` restricted to absolute frame indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = list(indices)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[self._indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self._indices)


def _unwrap_chain(dataset):
    seen = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        nxt = getattr(current, "dataset", None)
        if nxt is None:
            nxt = getattr(current, "_dataset", None)
        current = nxt


def _as_int(value) -> int:
    if isinstance(value, list | tuple):
        if len(value) != 1:
            raise ValueError(f"Expected a scalar episode field, got {value}")
        value = value[0]
    return int(value)


def _episode_rows(episodes) -> list[tuple[int, int, int]]:
    if episodes is None:
        return []
    if isinstance(episodes, list | tuple):
        rows = episodes
    else:
        try:
            indices = episodes["episode_index"]
            starts = episodes["dataset_from_index"]
            ends = episodes["dataset_to_index"]
        except (KeyError, TypeError, IndexError):
            rows = [episodes[index] for index in range(len(episodes))]
        else:
            return [
                (_as_int(episode), _as_int(start), _as_int(end))
                for episode, start, end in zip(indices, starts, ends, strict=True)
            ]
    return [
        (
            _as_int(row["episode_index"]),
            _as_int(row["dataset_from_index"]),
            _as_int(row["dataset_to_index"]),
        )
        for row in rows
    ]


def episode_frame_spans(dataset) -> list[tuple[int, int, int]] | None:
    """Return ``(episode_index, start_frame, end_frame)`` for a LeRobot-backed dataset.

    Indices are positions in the full dataset, so a later subset still lines up with
    wrappers that look up actions by the original frame index.
    """
    for current in _unwrap_chain(dataset):
        bounds = getattr(current, "episode_bounds", None)
        if bounds:
            return [(int(episode), int(start), int(end)) for episode, (start, end) in sorted(bounds.items())]
    for current in _unwrap_chain(dataset):
        meta = getattr(current, "meta", None)
        episodes = getattr(meta, "episodes", None) if meta is not None else None
        rows = _episode_rows(episodes)
        if rows:
            return sorted(rows, key=lambda item: item[0])
    return None


def split_episode_frames(
    spans: Sequence[tuple[int, int, int]], val_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Split whole episodes into train and validation frame indices.

    Episodes are ordered by index, shuffled with ``RandomState(seed)``, and the first
    ``n_val = max(1, int(len(spans) * val_ratio))`` episodes become validation. The two
    sides do not share episodes. Every frame of a selected episode is kept.
    """
    spans = sorted(spans, key=lambda item: item[0])
    episodes = [episode for episode, _, _ in spans]
    if len(set(episodes)) != len(episodes):
        raise ValueError("Episode indices must be unique")
    count = len(spans)
    if count < 2:
        raise ValueError(f"Need at least 2 episodes to hold out val_ratio={val_ratio}")
    n_val = min(count - 1, max(1, int(count * val_ratio)))
    order = np.random.RandomState(seed).permutation(count)
    val_episodes = [spans[int(index)][0] for index in order[:n_val]]
    train_episodes = [spans[int(index)][0] for index in order[n_val:]]
    val_ids = set(val_episodes)
    train_frames: list[int] = []
    val_frames: list[int] = []
    for episode, start, end in spans:
        if end < start:
            raise ValueError(f"Episode {episode} has a descending frame range {start}:{end}")
        frames = range(start, end)
        if episode in val_ids:
            val_frames.extend(frames)
        else:
            train_frames.extend(frames)
    return train_frames, val_frames, train_episodes, val_episodes


class FakeDataset(Dataset):
    """Deterministic NumPy fixtures for exercising the PyTorch training pipeline."""

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._model_config = model_config

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = np.random.default_rng(index.__index__())
        config = self._model_config

        def image():
            return rng.uniform(-1, 1, (*_model.IMAGE_RESOLUTION, 3)).astype(np.float32)

        return {
            "image": {key: image() for key in _model.IMAGE_KEYS},
            "image_mask": dict.fromkeys(_model.IMAGE_KEYS, np.True_),
            "state": rng.uniform(-1, 1, (config.action_dim,)).astype(np.float32),
            "tokenized_prompt": rng.integers(0, 2048, (config.max_token_len,), dtype=np.int32),
            "tokenized_prompt_mask": np.ones(config.max_token_len, dtype=bool),
            "actions": rng.uniform(-1, 1, (config.action_horizon, config.action_dim)).astype(np.float32),
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    root = resolve_dataset_root(repo_id)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    if data_config.align_franka_action_trace and not np.isclose(dataset_meta.fps, 30.0, atol=0.01):
        raise ValueError(f"ROS multirate training requires a 30 Hz LeRobot dataset, got {dataset_meta.fps} Hz")
    # Cameras with dtype "video" are decoded from videos/<key>/*.mp4 by LeRobot.
    # Retained collector datasets store one combined actions column instead of action.*.
    keys, legacy_actions = sequence_keys(dataset_meta.features, data_config.action_sequence_keys)
    delta_timestamps = (
        {}
        if data_config.align_franka_action_trace
        else {key: [t / dataset_meta.fps for t in range(action_horizon)] for key in keys}
    )
    delta_timestamps.update(
        {
            key: [delta / dataset_meta.fps for delta in delta_indices]
            for key, delta_indices in data_config.delta_indices.items()
        }
    )
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=dataset_meta.root,
        delta_timestamps=delta_timestamps or None,
        # Policies consume uint8 images; avoid converting decoded MP4 pixels to
        # float32 only to convert them back before preprocessing.
        return_uint8=True,
    )
    if legacy_actions:
        dataset = LegacyCollectorFrames(dataset)

    if data_config.align_franka_action_trace:
        dataset = MultirateFrankaDataset(dataset, action_horizon)

    if data_config.prompt_from_task:
        tasks = {int(index): str(task) for task, index in dataset_meta.tasks["task_index"].items()}
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    batch_size: int | None = None,
    dataset: Dataset | None = None,
    frame_indices: Sequence[int] | None = None,
    repeat: bool = True,
    drop_last: bool = True,
    use_distributed: bool = True,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = config.batch_size if batch_size is None else batch_size
    return create_torch_data_loader(
        data_config,
        config.model,
        config.model.action_horizon,
        batch_size,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        dataset=dataset,
        frame_indices=frame_indices,
        repeat=repeat,
        drop_last=drop_last,
        use_distributed=use_distributed,
    )


def create_train_and_val_loaders(
    config: _config.TrainConfig, *, batch_size: int
) -> tuple[DataLoader, DataLoader | None, _config.DataConfig, Dataset | None, list[int] | None]:
    """Build the training loader and, when episodes can be split, a one-pass validation loader.

    Both loaders share one transformed dataset, so they use the same ``norm_stats``.
    Validation frames are this rank's stride of the held-out episodes.
    Disabled validation uses the original loader and preview construction path.
    """
    if not config.eval_enabled or config.val_ratio == 0:
        loader = create_data_loader(config, shuffle=True, batch_size=batch_size)
        return loader, None, loader.data_config(), None, None

    data_config = config.data.create(config.assets_dirs, config.model)
    raw = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed = transform_dataset(raw, data_config)
    train_indices = None
    val_loader = None
    if config.val_ratio > 0:
        spans = episode_frame_spans(raw)
        if spans is None:
            if data_config.repo_id != "fake":
                logging.warning("Dataset has no episode index; training uses every frame and skips validation")
        else:
            train_indices, val_indices, train_episodes, val_episodes = split_episode_frames(
                spans, config.val_ratio, config.split_seed
            )
            world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                logging.info(
                    "Validation episodes (%d/%d, %d frames): %s. Training uses %d episodes (%d frames).",
                    len(val_episodes),
                    len(spans),
                    len(val_indices),
                    val_episodes,
                    len(train_episodes),
                    len(train_indices),
                )
            shard = val_indices[rank::world]
            local_batch = min(batch_size // world, len(shard))
            if shard and local_batch > 0:
                val_loader = create_torch_data_loader(
                    data_config,
                    config.model,
                    config.model.action_horizon,
                    local_batch,
                    dataset=transformed,
                    frame_indices=shard,
                    shuffle=False,
                    num_workers=config.num_workers,
                    seed=config.split_seed,
                    repeat=False,
                    drop_last=False,
                    use_distributed=False,
                    persistent_workers=False,
                )
    train_loader = create_torch_data_loader(
        data_config,
        config.model,
        config.model.action_horizon,
        batch_size,
        dataset=transformed,
        frame_indices=train_indices,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return train_loader, val_loader, data_config, transformed, train_indices


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    dataset: Dataset | None = None,
    frame_indices: Sequence[int] | None = None,
    repeat: bool = True,
    drop_last: bool = True,
    use_distributed: bool = True,
    persistent_workers: bool | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    if dataset is None:
        dataset = transform_dataset(
            create_torch_dataset(data_config, action_horizon, model_config),
            data_config,
            skip_norm_stats=skip_norm_stats,
        )
    if frame_indices is not None:
        dataset = SubsetDataset(dataset, frame_indices)
    sampler = None
    distributed = use_distributed and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    if batch_size <= 0 or batch_size % world_size:
        raise ValueError(f"Global batch_size={batch_size} must be positive and divisible by world_size={world_size}")
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
    loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // world_size,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        drop_last=drop_last,
        repeat=repeat,
        persistent_workers=persistent_workers,
    )
    return DataLoaderImpl(data_config, loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        repeat: bool = True,
        persistent_workers: bool | None = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
            drop_last: Drop a trailing partial batch. Validation keeps it.
            repeat: Start another epoch after the dataset is exhausted. Validation reads once.
            persistent_workers: Keep worker processes after an epoch. Defaults to on when
                ``num_workers > 0``.
        """
        if local_batch_size <= 0 or len(dataset) == 0 or (drop_last and len(dataset) < local_batch_size):
            raise ValueError(f"Local batch size ({local_batch_size}) is invalid for dataset size ({len(dataset)}).")
        self._num_batches = num_batches
        self._repeat = repeat

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0 if persistent_workers is None else persistent_workers,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        epoch = 0
        while True:
            if isinstance(self._data_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self._data_loader.sampler.set_epoch(epoch)
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield tree_map(torch.as_tensor, batch)
            epoch += 1
            if not self._repeat:
                return


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    return tree_map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    # Shared transforms import JAX; keep worker imports on CPU.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    np.random.seed(torch.initial_seed() % (2**32))


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
