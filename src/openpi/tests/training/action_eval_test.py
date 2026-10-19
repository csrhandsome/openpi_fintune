from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.franka_policy import ActionLayout
from openpi.training import config
from openpi.training.action_eval import component_slices
from openpi.training.action_eval import denoised_action_metrics
from openpi.training.action_eval import squared_error_totals
from openpi.training.vla import _log_validation


def test_component_slices_follow_the_arm_layout():
    joint = component_slices(ActionLayout("joint", 8, 7, 1))
    assert joint == {"joint": (0, 7), "gripper": (7, 8)}
    ee = component_slices(ActionLayout("ee", 14, 6, 1))
    assert ee["left_pos"] == (0, 3)
    assert ee["left_rot"] == (3, 6)
    assert ee["left_gripper"] == (6, 7)
    assert ee["right_pos"] == (7, 10)
    assert ee["right_rot"] == (10, 13)
    assert ee["right_gripper"] == (13, 14)


def test_squared_error_ignores_padding():
    prediction = torch.zeros(2, 4, 32)
    target = torch.ones(2, 4, 32)
    target[..., 8:] = 5
    target[..., 7] = 0
    totals = squared_error_totals(prediction, target, 8, {"joint": (0, 7), "gripper": (7, 8)})
    assert totals["loss"] == (2 * 4 * 7, 2 * 4 * 8)
    assert totals["joint"] == (2 * 4 * 7, 2 * 4 * 7)
    assert totals["gripper"] == (0.0, 2 * 4)


class _Sampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = None
        self.saw_training = None

    def sample_actions(self, observation, num_steps=10):
        assert not torch.is_grad_enabled()
        self.steps = num_steps
        self.saw_training = self.training
        return observation + 1


def test_eval_loss_is_denoised_action_mse():
    sampler = _Sampler()
    sampler.train()
    actions = torch.zeros(2, 3, 32)
    actions[..., :8] = 1
    actions[..., 8:] = 4
    metrics = denoised_action_metrics(
        sampler,
        [(actions, actions)],
        physical_dim=8,
        slices=component_slices(ActionLayout("joint", 8, 7, 1)),
        num_steps=10,
        device=torch.device("cpu"),
    )
    assert sampler.steps == 10
    assert sampler.saw_training is False
    assert sampler.training is True
    assert metrics["eval_loss"] == 1
    assert metrics["eval_l2_joint"] == 1
    assert metrics["eval_l2_gripper"] == 1


def test_eval_weights_partial_batches_by_element_count():
    sampler = _Sampler()
    target = torch.zeros(3, 50, 32)
    observation = torch.zeros_like(target)
    observation[-1, :, :8] = 2  # Errors: 1, 1, 9; the final batch has one sample.
    metrics = denoised_action_metrics(
        sampler,
        [(observation[:2], target[:2]), (observation[2:], target[2:])],
        physical_dim=8,
        slices=component_slices(ActionLayout("joint", 8, 7, 1)),
        num_steps=10,
        device="cpu",
    )
    assert metrics == pytest.approx({"eval_loss": 11 / 3, "eval_l2_joint": 11 / 3, "eval_l2_gripper": 11 / 3})


@pytest.mark.parametrize("training", [True, False])
def test_eval_restores_model_mode_after_sampling_failure(training):
    sampler = _Sampler().train(training)
    sampler.sample_actions = Mock(side_effect=RuntimeError("sampler failed"))
    with pytest.raises(RuntimeError, match="sampler failed"):
        denoised_action_metrics(sampler, [(None, None)], physical_dim=8, slices={}, num_steps=10, device="cpu")
    assert sampler.training is training


def test_euler_integrates_ten_steps_to_the_denoised_action():
    observation = SimpleNamespace(state=torch.zeros(2, 32))
    target = torch.full((2, 50, 32), 0.25)
    noise = torch.randn_like(target)
    times = []

    def suffix(_observation, _actions, time, _cache, _mask):
        times.append(time[0].item())
        return noise - target  # Exact velocity for this straight flow path.

    sampler = SimpleNamespace(run_suffix=suffix, velocity_from_suffix=lambda hidden: hidden)
    prediction = PI0Pytorch._sample_actions_from_cache(sampler, observation, noise, None, None, 10)  # noqa: SLF001
    assert times == pytest.approx([1 - step / 10 for step in range(10)])
    torch.testing.assert_close(prediction, target, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("data", "physical_dim", "components"),
    [
        (
            config.LeRobotFrankaDataConfig(action_space="ee", action_dim=7),
            7,
            ("pos", "rot", "gripper"),
        ),
        (
            config.LeRobotFrankaDataConfig(action_space="ee_delta", action_dim=7),
            7,
            ("pos", "rot", "gripper"),
        ),
        (
            config.LeRobotFrankaDataConfig(action_space="joint", action_dim=8),
            8,
            ("joint", "gripper"),
        ),
        (
            config.LeRobotFrankaDataConfig(action_space="ee_delta", action_dim=14),
            14,
            ("left_pos", "left_rot", "left_gripper", "right_pos", "right_rot", "right_gripper"),
        ),
    ],
    ids=["ee", "ee-delta", "joint", "dual-ee"],
)
def test_validation_uses_physical_layout(data, physical_dim, components):
    settings = config.TrainConfig(name="validation_test", data=data)
    observation = torch.zeros(2, 3, 32)
    observation[..., physical_dim:] = 9
    target = torch.ones_like(observation)
    target[..., physical_dim:] = 0
    tracker = Mock()
    sampler = _Sampler()

    _log_validation(
        sampler,
        [(observation, target)],
        settings,
        device=torch.device("cpu"),
        dtype=torch.float32,
        step=500,
        is_main=True,
        tracker=tracker,
    )

    tracker.log.assert_called_once()
    metrics = tracker.log.call_args.args[0]
    assert tracker.log.call_args.kwargs == {"step": 500}
    assert metrics == {"eval_loss": 0.0, **{f"eval_l2_{name}": 0.0 for name in components}}
    assert sampler.steps == settings.eval_num_steps
    assert sampler.training is True
