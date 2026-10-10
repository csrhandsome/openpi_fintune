"""Offline checks for contact synchronization, safety and E-step TD semantics."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from openpi.rlt.force_rlt.action_adapter import FrankaSafetyProjector
from openpi.rlt.force_rlt.action_adapter import SafetyLimits
from openpi.rlt.force_rlt.action_adapter import reanchor_reference
from openpi.rlt.force_rlt.learner import ChunkLearner
from openpi.rlt.force_rlt.learner import LearnerConfig
from openpi.rlt.force_rlt.learner import td_target
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.rlt.force_rlt.observation import FingerFrame
from openpi.rlt.force_rlt.observation import FingerHistory
from openpi.rlt.force_rlt.raw_frames import RawFrameJournal
from openpi.rlt.force_rlt.replay import ReplayJournal
from openpi.rlt.force_rlt.replay import State
from openpi.rlt.force_rlt.replay import Transition


class TinyEncoder:
    feature_dim = 2

    def __call__(self, left, right):
        return left.astype(np.float32).reshape(2), right.astype(np.float32).reshape(2)


def test_finger_order_history_and_stale_frame():
    history = FingerHistory(TinyEncoder(), max_skew_s=0.01, max_age_s=0.1)
    left = FingerFrame("left0", 1.0, np.array([1, 2], np.uint8))
    right = FingerFrame("right0", 1.001, np.array([3, 4], np.uint8))
    history.start_episode(left, right)
    value = history.append(
        FingerFrame("left1", 1.05, np.array([2, 4], np.uint8)),
        FingerFrame("right1", 1.051, np.array([5, 8], np.uint8)),
        robot_ts=1.06,
    )
    np.testing.assert_array_equal(value.z_grip[-12:], [2, 4, 1, 2, 1, 2, 5, 8, 2, 4, 2, 4])
    assert value.valid_mask.tolist() == [False, False, False, True]
    stale = history.append(FingerFrame("left2", 1.1, None), right, robot_ts=1.3)
    assert not stale.valid_mask[-1]
    assert np.all(stale.z_grip[-12:] == 0)


class FakeKinematics:
    def fk(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def ik(self, pose, seed):
        q = seed.copy()
        q[:3] = pose[:3, 3]
        return q


def limits():
    return SafetyLimits(
        joint_lower=np.full(7, -2.0),
        joint_upper=np.full(7, 2.0),
        max_joint_step=np.full(7, 0.2),
        max_joint_velocity=np.full(7, 10.0),
        max_joint_acceleration=np.full(7, 500.0),
        max_tcp_step_xyz=np.full(3, 0.2),
        max_tcp_velocity_xyz=np.full(3, 10.0),
        max_tcp_acceleration_xyz=np.full(3, 500.0),
        max_tcp_angle_step_rad=0.2,
        max_tcp_angular_velocity_rad_s=10.0,
        max_tcp_angular_acceleration_rad_s2=500.0,
        workspace_lower_xyz=np.full(3, -2.0),
        workspace_upper_xyz=np.full(3, 2.0),
        reference_radius=np.full(7, 0.05),
        control_hz=30.0,
        max_ref_age_s=0.2,
    )


def test_reanchor_projection_and_stale_fallback():
    kinematics = FakeKinematics()
    ref = np.zeros((4, 7))
    ref[:, 0] = [0.01, 0.02, 0.03, 0.04]
    current = np.zeros(7)
    current[0] = 0.1
    anchored = reanchor_reference(ref, np.eye(4), kinematics.fk(current), current, kinematics)
    np.testing.assert_allclose(anchored[:, 0], [0.11, 0.12, 0.13, 0.14])
    projector = FrankaSafetyProjector(limits(), kinematics, prediction_horizon=4, execution_horizon=2)
    kwargs = {
        "current_q": current,
        "current_tcp": kinematics.fk(current),
        "prior_velocity": np.zeros(7),
        "prior_tcp_velocity_xyz": np.zeros(3),
        "prior_tcp_angular_speed": 0.0,
        "ref_anchor_tcp": np.eye(4),
        "ref_obs_ts": 1.0,
        "ref_generation_ts": 100.0,
        "now": 1.1,
    }
    accepted = projector.project(ref, ref, **kwargs)
    assert accepted.accepted
    np.testing.assert_allclose(accepted.executed_q[:, 0], [0.11, 0.12])
    rejected = projector.project(ref, ref, **{**kwargs, "now": 2.0})
    assert not rejected.accepted
    assert rejected.flags == ("stale_reference",)
    np.testing.assert_array_equal(rejected.executed_q, np.repeat(current[None], 2, axis=0))


def record(cfg, *, done=False, count=2, source="RL"):
    state = State(
        np.ones(cfg.z_rlt_dim, np.float32),
        np.ones(cfg.z_grip_dim, np.float32),
        np.zeros(7, np.float32),
        np.zeros((cfg.horizon, 7), np.float32),
    )
    mask = np.arange(cfg.executed_steps) < count
    rewards = np.ones(cfg.executed_steps, np.float32) * mask
    return Transition(
        "ep1",
        0,
        state,
        np.zeros((cfg.horizon, 7), np.float32),
        np.zeros((cfg.executed_steps, 7), np.float32),
        mask,
        rewards,
        done,
        state,
        source,
        "v1",
        (),
        independent_success=False,
        intervention_flag=source == "HUMAN",
        observation_meta={},
    )


def test_partial_execution_target_and_human_isolation(tmp_path):
    cfg = NetworkConfig(z_rlt_dim=4, z_grip_dim=8, horizon=4, executed_steps=2, hidden_dim=16)
    rewards = torch.tensor([[1.0, 9.0], [1.0, 2.0]])
    mask = torch.tensor([[True, False], [True, True]])
    target = td_target(rewards, mask, torch.tensor([True, False]), torch.tensor([100.0, 3.0]), 0.9)
    torch.testing.assert_close(target, torch.tensor([1.0, 1 + 1.8 + 0.81 * 3]))
    journal = ReplayJournal(tmp_path / "replay.jsonl", cfg)
    transition = record(cfg, done=True, count=1)
    journal.append(transition)
    assert len(journal.read_all()) == 1
    learner = ChunkLearner(LearnerConfig(cfg))
    metrics = learner.update([transition])
    assert np.isfinite(metrics["actor_loss"])
    learner.save(tmp_path / "actor.pt")
    loaded = ChunkLearner.load(tmp_path / "actor.pt")
    assert loaded.steps == 1
    with pytest.raises(ValueError, match="Human"):
        learner.update([record(cfg, done=True, source="HUMAN")])
    with pytest.raises(ValueError, match="short execution"):
        replace(transition, done=False).validate(cfg)


def test_raw_frame_journal_preserves_rgb(tmp_path):
    journal = RawFrameJournal(tmp_path, "episode_1")
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    path = journal.append(-1, finger_left=frame, finger_right=None)
    journal.close()
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive["finger_left"], frame)
        assert "finger_right" not in archive
