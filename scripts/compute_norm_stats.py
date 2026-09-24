"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import os
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
from openpi.policies import audiovla_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class AudioNormStatsInputs(transforms.DataTransformFn):
    """Build only the fields needed for state/action norm stats."""

    def __init__(self, action_dim: int):
        self._action_dim = action_dim

    def __call__(self, data: dict) -> dict:
        joint_position = data.get("observation/joint_position", data.get("joint_position"))
        gripper_position = data.get("observation/gripper_position", data.get("gripper_position"))
        if joint_position is not None and gripper_position is not None:
            gripper_position = np.asarray(gripper_position, dtype=np.float32)
            if gripper_position.ndim == 0:
                gripper_position = gripper_position[np.newaxis]
            state = np.concatenate(
                [
                    np.asarray(joint_position, dtype=np.float32),
                    gripper_position,
                ],
                axis=-1,
            )
        else:
            state = data.get("observation/state", data.get("state"))
            if state is None:
                raise KeyError("Missing state fields for norm stats.")

        output = {"state": transforms.pad_to_dim(np.asarray(state, dtype=np.float32), self._action_dim)}

        actions = data.get("actions", data.get("action"))
        if actions is not None:
            output["actions"] = transforms.pad_to_dim(np.asarray(actions, dtype=np.float32), self._action_dim)
        return output


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _norm_stats_transforms(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    data_transforms = []
    for transform in data_config.data_transforms.inputs:
        # Norm stats use only robot state/action.  Do not decode dialogue
        # sidecars or Qwen features: these are unrelated to normalization and
        # make a CPU-only stats pass needlessly expensive.
        if isinstance(transform, transforms.ExtractDialogueTurn):
            continue
        if isinstance(transform, audiovla_policy.AudiovlaInputs):
            data_transforms.append(AudioNormStatsInputs(transform.action_dim))
        elif isinstance(transform, transforms.AudioVLAConditioning):
            continue
        else:
            data_transforms.append(transform)

    return [
        *data_config.repack_transforms.inputs,
        *data_transforms,
        # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
        RemoveStrings(),
    ]


def compute_omniaction_libero_norm_stats(
    config: _config.TrainConfig,
    *,
    max_frames: int | None,
) -> dict[str, normalize.NormStats]:
    """Compute OmniAction-LIBERO stats directly from parquet state/action columns.

    The usual LeRobot data path decodes both camera images before the norm-stat
    transforms discard them.  That is unnecessary for state/action statistics,
    expensive, and can stall a CPU-only job.  Each OmniAction episode is one
    parquet file, so action chunks can be recreated exactly by clamping the
    future-action horizon at its episode boundary.
    """
    if not isinstance(config.data, _config.LeRobotOmniActionLiberoDataConfig):
        raise TypeError("Expected an OmniAction-LIBERO data config.")
    if config.data.repo_id is None:
        raise ValueError("OmniAction-LIBERO data config must have a repo_id.")

    # Keep compatibility with the project convention (HF_LEROBOT_HOME) and
    # LeRobot's conventional variable (LEROBOT_HOME).
    lerobot_home = pathlib.Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            os.environ.get("LEROBOT_HOME", "~/.cache/huggingface/lerobot"),
        )
    ).expanduser()
    dataset_root = lerobot_home / config.data.repo_id
    data_root = dataset_root / "data"
    parquet_files = sorted(data_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet episodes found under {data_root}")

    state_dim = config.model.action_dim
    action_horizon = config.model.action_horizon
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive.")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    processed_frames = 0
    for parquet_path in tqdm.tqdm(parquet_files, desc="Computing OmniAction stats"):
        table = pq.read_table(parquet_path, columns=["state", "actions"])
        state = np.asarray(table.column("state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
        if state.ndim != 2 or actions.ndim != 2 or len(state) != len(actions):
            raise ValueError(f"Invalid state/actions layout in {parquet_path}")
        if len(state) == 0:
            continue

        remaining = None if max_frames is None else max_frames - processed_frames
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None:
            state = state[:remaining]
            actions = actions[:remaining]
        num_frames = len(state)

        # This is equivalent to LeRobotDataset's action delta-timestamp query:
        # actions[t : t + action_horizon], with the last available action
        # repeated after the episode boundary.
        future_indices = np.minimum(
            np.arange(num_frames)[:, None] + np.arange(action_horizon)[None, :],
            num_frames - 1,
        )
        action_chunks = actions[future_indices]

        padded_state = transforms.pad_to_dim(state, state_dim)
        padded_actions = transforms.pad_to_dim(action_chunks, state_dim)

        # Match the OmniAction data transform: the first six action channels
        # are delta actions, while the seventh gripper channel is absolute.
        padded_actions[..., :6] -= padded_state[:, None, :6]

        stats["state"].update(padded_state)
        stats["actions"].update(padded_actions)
        processed_frames += num_frames

    if processed_frames == 0:
        raise ValueError("No OmniAction-LIBERO frames were available for norm stats.")
    return {key: value.get_statistics() for key, value in stats.items()}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # Text-only Talker samples deliberately oversample a few dialogue frames;
    # they must not change state/action normalization statistics.
    data_config = dataclasses.replace(data_config, dialogue_turn_augmentation=False)
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(dataset, _norm_stats_transforms(data_config))
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        _norm_stats_transforms(data_config),
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    batch_size: int | None = None,
    qwen: bool = False,
    qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B",
    qwen_device: str = "auto",
):
    config = _config.get_config(config_name)
    if qwen:
        config = _config.enable_qwen_encoder(
            config,
            qwen_model_name=qwen_model_name,
            qwen_device=qwen_device,
        )
    data_config = config.data.create(config.assets_dirs, config.model)

    effective_batch_size = batch_size if batch_size is not None else config.batch_size

    if isinstance(config.data, _config.LeRobotOmniActionLiberoDataConfig):
        norm_stats = compute_omniaction_libero_norm_stats(config, max_frames=max_frames)
        output_path = config.assets_dirs / config.data.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, effective_batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
