import dataclasses
from unittest.mock import Mock

import jax
import numpy as np
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


@pytest.mark.parametrize(("enabled", "ratio"), [(False, 0.05), (True, 0.0)])
def test_disabled_eval_uses_original_loader_without_inspecting_episodes(monkeypatch, enabled, ratio):
    settings = _config.TrainConfig(name="no_eval", eval_enabled=enabled, val_ratio=ratio)
    loader = Mock()
    create = Mock(return_value=loader)
    inspect_episodes = Mock(side_effect=AssertionError("Disabled evaluation must not inspect episodes"))
    monkeypatch.setattr(_data_loader, "create_data_loader", create)
    monkeypatch.setattr(_data_loader, "episode_frame_spans", inspect_episodes)
    result = _data_loader.create_train_and_val_loaders(settings, batch_size=4)
    create.assert_called_once_with(settings, shuffle=True, batch_size=4)
    inspect_episodes.assert_not_called()
    assert result == (loader, None, loader.data_config(), None, None)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_distributed_sampler_reshuffles_without_overlapping_ranks():
    dataset = list(range(16))
    rank_batches = []
    for rank in range(2):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=2, rank=rank, seed=42, shuffle=True
        )
        loader = _data_loader.TorchDataLoader(dataset, local_batch_size=1, sampler=sampler, num_batches=16)
        rank_batches.append([int(batch.item()) for batch in loader])
    for epoch in range(2):
        start = epoch * 8
        left, right = (set(batches[start : start + 8]) for batches in rank_batches)
        assert left.isdisjoint(right)
        assert left | right == set(dataset)
    assert rank_batches[0][:8] != rank_batches[0][8:]


def test_validation_reads_one_epoch_and_keeps_the_partial_batch():
    class _Rows:
        def __len__(self):
            return 5

        def __getitem__(self, index):
            return np.int64(index)

    loader = _data_loader.TorchDataLoader(_Rows(), local_batch_size=2, drop_last=False, repeat=False)
    batches = [int(batch.sum()) for batch in loader]
    assert batches == [1, 5, 4]


def test_episode_split_holds_out_five_percent_of_files():
    spans = [(episode, episode * 10, (episode + 1) * 10) for episode in range(200)]
    train_frames, val_frames, train_episodes, val_episodes = _data_loader.split_episode_frames(spans, 0.05, 42)
    assert len(val_episodes) == 10
    assert len(train_episodes) == 190
    assert set(train_episodes).isdisjoint(val_episodes)
    assert set(train_episodes) | set(val_episodes) == set(range(200))
    assert set(train_frames).isdisjoint(val_frames)
    assert len(val_frames) == 100
    assert _data_loader.split_episode_frames(spans, 0.05, 42)[3] == val_episodes

    short = [(0, 0, 1), (1, 1, 50)]
    _, val_frames, _, val_episodes = _data_loader.split_episode_frames(short, 0.05, 0)
    assert len(val_episodes) == 1
    assert val_frames == ([0] if val_episodes == [0] else list(range(1, 50)))
    with pytest.raises(ValueError, match="at least 2"):
        _data_loader.split_episode_frames(spans[:1], 0.05, 42)


def test_episode_spans_follow_wrappers_and_sort_by_index():
    class _Meta:
        def __init__(self):
            self.episodes = [
                {"episode_index": 1, "dataset_from_index": 4, "dataset_to_index": 6},
                {"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 4},
            ]

    class _Inner:
        def __init__(self):
            self.meta = _Meta()

    class _Wrapped:
        def __init__(self):
            self._dataset = _Inner()

    assert _data_loader.episode_frame_spans(_Wrapped()) == [(0, 0, 4), (1, 4, 6)]

    class _Bounded:
        def __init__(self):
            self.episode_bounds = {1: (5, 8), 0: (0, 5)}
            self.dataset = _Inner()

    assert _data_loader.episode_frame_spans(_Bounded()) == [(0, 0, 5), (1, 5, 8)]


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = dataclasses.replace(_config.get_config("openpi"), data=_config.FakeDataConfig())

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for observation, actions in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves((observation.to_dict(), actions)))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


@pytest.mark.manual
def test_with_real_dataset():
    config = dataclasses.replace(
        _config.get_config("openpi"),
        data=_config.LeRobotFrankaDataConfig(repo_id="robot/local_test", action_space="ee_delta", action_dim=7),
    )
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
