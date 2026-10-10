"""Exercise masked TD, independent networks, action layouts and checkpoint continuation."""

from dataclasses import replace
import sys

import numpy as np
import pytest
import torch

from openpi.rlt import train
from openpi.rlt.action_representation import ActionAdapter
from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.learner import ChunkLearner
from openpi.rlt.learner import LearnerConfig
from openpi.rlt.learner import td_target
from openpi.rlt.replay import ReplayJournal
from openpi.tests.rlt.replay_test import episode


def test_td_does_not_bootstrap_terminal_or_padding():
    target = td_target(
        torch.tensor([[1.0, 99.0], [0.0, 1.0]]),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([True, False]),
        torch.tensor([float("nan"), 2.0]),
        0.9,
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 0.9 + 0.81 * 2]))


@pytest.mark.parametrize(
    ("space", "dimension", "representation"), [("ee", 7, "ee_delta"), ("joint", 8, "absolute"), ("ee", 14, "ee_delta")]
)
def test_action_adapter_roundtrip(space, dimension, representation):
    del space
    adapter = ActionAdapter(
        dimension, representation, np.full(dimension, -2, np.float32), np.full(dimension, 2, np.float32)
    )
    state = np.full((2, dimension), 0.2, np.float32)
    actions = np.full((2, 3, dimension), 0.3, np.float32)
    normalized = adapter.normalize(actions, state)
    np.testing.assert_allclose(adapter.denormalize(normalized, state), actions, atol=1e-6)
    if representation == "ee_delta":
        assert normalized[0, 0, 0] != normalized[0, 0, 6]


def test_learner_human_short_episode_checkpoint_and_proposal(tmp_path):
    spec = ReplaySpec(4, 7, 4, 3, 7)
    record = episode(spec, length=1)[0]
    sources = record.source_chunk.copy()
    sources[0] = "HUMAN"
    record = replace(record, source_chunk=sources)
    learner = ChunkLearner(
        spec,
        LearnerConfig(actor_update_period=1),
        actor_config=ActorConfig(hidden_dim=8, layers=1),
        critic_config=CriticConfig(hidden_dim=8, layers=1),
    )
    before = [p.detach().clone() for p in learner.actor.parameters()]
    metrics = learner.update([record])
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.actor.parameters(), strict=True))
    assert all(p.requires_grad for p in learner.critic.parameters())
    path = tmp_path / "actor_critic.pt"
    learner.save(path)
    restored = ChunkLearner.load(path)
    assert restored.steps == 1
    feature = {"z_rlt": record.z_rlt, "proprio": record.proprio, "ref_chunk": record.ref_chunk}
    np.testing.assert_allclose(learner.propose(feature), restored.propose(feature))
    assert restored.propose(feature).shape == (4, 7)
    assert np.isfinite(restored.update([record])["critic_loss"])


def test_offline_training_cli_and_resume(tmp_path, monkeypatch):
    spec = ReplaySpec(4, 7, 4, 3, 7)
    replay = tmp_path / "replay"
    journal = ReplayJournal(replay, spec)
    journal.replace_episode(episode(spec), metadata={"success": True})
    output = tmp_path / "learner.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--replay",
            str(replay),
            "--output",
            str(output),
            "--steps",
            "2",
            "--batch-size",
            "2",
            "--actor-hidden-dim",
            "8",
            "--critic-hidden-dim",
            "8",
        ],
    )
    train.main()
    assert ChunkLearner.load(output).steps == 2
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--replay",
            str(replay),
            "--output",
            str(output),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--resume",
            str(output),
        ],
    )
    train.main()
    assert ChunkLearner.load(output).steps == 3


def test_reference_optimizer_targets_q_loss_and_physical_smoothing():
    spec = ReplaySpec(4, 7, 4, 3, 7)
    record = episode(spec)[0]
    learner = ChunkLearner(
        spec,
        LearnerConfig(delta_weight=2.0, reference_dropout=0.0),
        actor_config=ActorConfig(hidden_dim=8, fixed_std=0.0),
        critic_config=CriticConfig(hidden_dim=8),
        adapter=ActionAdapter(7, q01=np.full(7, -2, np.float32), q99=np.full(7, 2, np.float32)),
    )
    assert type(learner.actor_opt) is torch.optim.Adam
    assert learner.actor_opt.param_groups[0]["weight_decay"] == 0.0
    before = [p.clone() for p in learner.target_critic.parameters()]
    learner.update([record])
    for expected, actual in zip(before, learner.target_critic.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    metrics = learner.update([record])
    assert metrics["delta_penalty"] > 0
    assert metrics["actor_loss"] == pytest.approx(
        metrics["bc"] - metrics["actor_q"] + 2 * metrics["delta_penalty"], abs=1e-5
    )
    for initial, target, current in zip(
        before, learner.target_critic.parameters(), learner.critic.parameters(), strict=True
    ):
        torch.testing.assert_close(target, initial.lerp(current, learner.config.tau), rtol=0, atol=0)
    for representation in ("absolute", "ee_delta"):
        adapter = ActionAdapter(7, representation, np.full(7, -2, np.float32), np.full(7, 2, np.float32))
        normalized = torch.randn(2, 3, 7, requires_grad=True)
        state = torch.randn(2, 7)
        physical = adapter.denormalize_tensor(normalized, state)
        np.testing.assert_allclose(
            physical.detach().numpy(), adapter.denormalize(normalized.detach().numpy(), state.numpy()), atol=1e-6
        )
        physical.sum().backward()
        assert torch.isfinite(normalized.grad).all()
