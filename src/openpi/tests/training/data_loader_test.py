import dataclasses

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


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
        _config.get_config("openpi"), data=_config.LeRobotFrankaEEDataConfig(repo_id="robot/local_test")
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
