"""The RL-token service returns physical reference and tokens from one VLA call."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.policies.policy import Policy


class FakeVLA(torch.nn.Module):
    action_horizon = 4
    action_dim = 32

    def __init__(self):
        super().__init__()
        self.llm = SimpleNamespace(configs=[SimpleNamespace(width=8)])
        self.calls = 0

    def sample_actions_with_prefix(self, observation, **_kwargs):
        self.calls += 1
        actions = torch.arange(4 * 32, dtype=torch.float32).reshape(1, 4, 32)
        prefix = torch.ones(1, 3, 8)
        mask = torch.tensor([[True, False, True]])
        return actions, prefix, mask

    def sample_actions(self, observation, **kwargs):
        return self.sample_actions_with_prefix(observation, **kwargs)[0]


class FakeReadout(torch.nn.Module):
    config = SimpleNamespace(input_dim=8)

    def encode(self, prefix, mask):
        return (prefix * mask[..., None]).sum(dim=1, keepdim=True)


@pytest.mark.parametrize(("action_space", "action_dim"), [("joint", 8), ("ee", 7)])
def test_physical_reference_and_token_share_one_call(action_space, action_dim):
    model = FakeVLA()
    policy = Policy(
        model,
        transforms=(),
        output_transforms=(lambda data: {"actions": data["actions"][:, :action_dim]},),
        pytorch_device="cpu",
        is_pytorch=True,
        metadata={"action_space": action_space},
    )
    policy.attach_rl_readout(FakeReadout(), metadata={"prefix_shape": [3, 8]})
    observation = {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        "state": np.zeros(8, np.float32),
        "monotonic_ts": np.array(10.0),
        "O_T_TCP": np.eye(4, dtype=np.float32),
    }
    result = policy.infer(observation)
    assert model.calls == 1
    assert result["ref_chunk"].shape == (4, action_dim)
    np.testing.assert_array_equal(result["ref_chunk"], result["actions"])
    if action_space == "joint":
        assert result["ref_q"].shape == (4, 7)
        np.testing.assert_array_equal(result["ref_q"], result["actions"][:, :7])
    else:
        assert "ref_q" not in result
    np.testing.assert_array_equal(result["z_rlt"], np.full(8, 2.0))
    assert float(result["ref_obs_ts"]) == 10.0
