import numpy as np
from openpi_client import action_chunk_broker
import pytest
import torch

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
    config = _config.get_config("openpi")
    policy = _policy_config.create_trained_policy(config, str(config.checkpoint_dir / "10000"))

    example = {
        "observation.exterior_image": np.zeros((224, 224, 3), np.uint8),
        "observation.wrist_image_left": np.zeros((224, 224, 3), np.uint8),
        "observation.ee_pose": np.zeros(6, np.float32),
        "observation.gripper_position": np.zeros(1, np.float32),
        "prompt": "test",
    }
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 7)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("openpi")
    policy = _policy_config.create_trained_policy(config, str(config.checkpoint_dir / "10000"))

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = {
        "observation.exterior_image": np.zeros((224, 224, 3), np.uint8),
        "observation.wrist_image_left": np.zeros((224, 224, 3), np.uint8),
        "observation.ee_pose": np.zeros(6, np.float32),
        "observation.gripper_position": np.zeros(1, np.float32),
        "prompt": "test",
    }
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
