from openpi_client import action_chunk_broker
import numpy as np
import pytest
import torch

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_pytorch_policy_calls_sampling_with_observation():
    class TorchModel(torch.nn.Module):
        def sample_actions(self, observation):
            assert isinstance(observation.state, torch.Tensor)
            return torch.zeros(1, 2, 32)

    policy = _policy.Policy(TorchModel(), is_pytorch=True, pytorch_device="cpu")
    obs = {
        "image": {},
        "image_mask": {},
        "state": np.zeros(32, dtype=np.float32),
    }
    result = policy.infer(obs)
    assert result["actions"].shape == (2, 32)


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
