"""Verify execution feedback, sparse rewards and episode-boundary actor handoff."""

import asyncio
import dataclasses
import json

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.rlt_client_policy import RLTClientPolicy
import pytest
import torch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
import yaml

from openpi.rlt.actor import ActorConfig
from openpi.rlt.critic import CriticConfig
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.learner import ChunkLearner
from openpi.rlt.online import OnlineLearner
from openpi.rlt.replay import ReplayJournal
from openpi.rlt.settings import CollectConfig
from openpi.rlt.settings import ForceConfig
from openpi.rlt.settings import LearnerSettings
from openpi.rlt.settings import ReplayConfig
from openpi.rlt.settings import WarmupConfig
from openpi.serving import config as inference_config
from openpi.serving import runner as server_runner
from openpi.serving.config import RLTConfig
from openpi.serving.rlt_policy import OnlineRLTPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as training_config
from openpi.training import runner as training_runner


class FrozenVLA:
    def __init__(self):
        self.metadata = {"action_space": "ee", "action_dim": 7, "action_horizon": 4, "z_rlt_dim": 4}
        self.calls = 0

    def infer(self, observation):
        assert not any(key.startswith("finger_") or key == "z_grip" for key in observation)
        self.calls += 1
        reference = np.full((4, 7), 0.2 + observation["stamp"] * 0.001, np.float32)
        return {"z_rlt": np.full(4, observation["stamp"], np.float32), "ref_chunk": reference, "actions": reference}


def observation(stamp, *, force=False):
    value = {"stamp": stamp, "observation.proprio": np.zeros(7, np.float32)}
    if force:
        value["z_grip"] = np.full(3, stamp, np.float32)
    return value


def make_policy(tmp_path, *, force=False, critical=False, start=False, freeze=False):
    settings = RLTConfig(
        chunk_len=2,
        deterministic=True,
        critical_phase_only=critical,
        warmup=WarmupConfig(min_transitions=2, updates=2),
        force=ForceConfig(enabled=force, input="features", feature_dim=3),
        replay=ReplayConfig(str(tmp_path / "replay"), capacity=20),
        learner=LearnerSettings(
            batch_size=2,
            actor_update_period=1,
            publish_interval=7,
            checkpoint_interval=2,
            checkpoint_path=str(tmp_path / "actor.pt"),
            freeze_after_warmup=freeze,
        ),
        actor=ActorConfig(hidden_dim=8, layers=1, fixed_std=0),
        critic=CriticConfig(hidden_dim=8, layers=1),
    )
    spec = ReplaySpec(4, 7, 4, 2, 7, grip_dim=settings.force.grip_dim)
    learner = ChunkLearner(
        spec, settings.learner.training_config(), actor_config=settings.actor, critic_config=settings.critic
    )
    journal = ReplayJournal(settings.replay.directory, spec)
    service = OnlineLearner(learner, journal, settings, start=start)
    return OnlineRLTPolicy(FrozenVLA(), service, settings), settings


def rpc(policy, event, episode_id="first", **values):
    return policy.infer({"rlt_event": event, "episode_id": episode_id, **values})


def collect_episode(policy, episode_id="first", *, success=True, force=False, partial=True):
    assert rpc(policy, "begin_episode", episode_id)["accepted"]
    first = rpc(policy, "infer", episode_id, observation=observation(0, force=force), in_critical_phase=True)
    actual = np.full((2, 7), 0.4, np.float32)
    rpc(
        policy,
        "observe",
        episode_id,
        step_id=first["step_id"],
        executed_actions=actual,
        next_observation=observation(1, force=force),
    )
    second = rpc(policy, "infer", episode_id, in_critical_phase=True)
    rpc(
        policy,
        "observe",
        episode_id,
        step_id=second["step_id"],
        executed_actions=actual[:1] if partial else actual,
        next_observation=observation(2, force=force),
    )
    rpc(policy, "end_episode", episode_id, success=success)
    return first, second


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("force", [True, False])
def test_collection_reward_feedback_and_handoff(tmp_path, success, force):
    policy, _ = make_policy(tmp_path, force=force)
    first, second = collect_episode(policy, success=success, force=force)
    assert first["source"] == second["source"] == "BASE"
    assert policy.vla_policy.calls == 3  # Cached reference is also the next-state reference.
    records = policy.service.journal.read_all()
    assert len(records) == 2
    np.testing.assert_array_equal(records[0].action_chunk, np.full((2, 7), 0.4, np.float32))
    np.testing.assert_array_equal(records[0].next_ref_chunk, records[1].ref_chunk)
    assert not records[0].done
    assert records[1].done
    np.testing.assert_array_equal(records[1].rewards, [int(success), 0])
    np.testing.assert_array_equal(records[1].executed_mask, [True, False])
    if force:
        np.testing.assert_array_equal(records[1].z_grip, [1, 1, 1])
        np.testing.assert_array_equal(records[1].next_z_grip, [2, 2, 2])
    assert not rpc(policy, "begin_episode", "second")["accepted"]
    assert policy.service.train_once()
    assert not policy.service.status()["ready_for_online"]
    assert policy.service.train_once()
    assert policy.service.status()["ready_for_online"]
    assert policy.service.status()["published_step"] == 2  # publish_interval=7
    assert rpc(policy, "begin_episode", "second")["phase"] == "online"
    response = rpc(policy, "infer", "second", observation=observation(3, force=force))
    assert response["source"] == "RL"
    assert response["actor_version"] >= 0
    assert not np.allclose(response["actions"], response["ref_chunk"])


def test_phase_stays_fixed_and_critical_or_base_selection(tmp_path):
    policy, _ = make_policy(tmp_path, critical=True)
    collect_episode(policy)
    # Simulate an episode already executing while the learner finishes.
    policy.phase.episode_phase = "warmup"
    policy.episode_id = "running"
    policy.service.train_once()
    policy.service.train_once()
    base = rpc(policy, "infer", "running", observation=observation(3), in_critical_phase=True)
    assert base["episode_phase"] == "warmup"
    assert base["source"] == "BASE"
    rpc(policy, "abort_episode", "running")
    assert rpc(policy, "begin_episode", "next")["phase"] == "online"
    base = rpc(policy, "infer", "next", observation=observation(4), in_critical_phase=False)
    assert base["source"] == "BASE"
    rpc(
        policy,
        "observe",
        "next",
        step_id=base["step_id"],
        executed_actions=base["actions"][:2],
        next_observation=observation(5),
    )
    actor = rpc(policy, "infer", "next", in_critical_phase=True)
    assert actor["source"] == "RL"
    rpc(policy, "abort_episode", "next")
    rpc(policy, "begin_episode", "base_eval", policy_mode="base")
    assert rpc(policy, "infer", "base_eval", observation=observation(6), in_critical_phase=True)["source"] == "BASE"


def test_feedback_validation_unannotated_abort_and_missing_force(tmp_path):
    policy, _ = make_policy(tmp_path, force=True)
    rpc(policy, "begin_episode")
    with pytest.raises(ValueError, match="z_grip"):
        rpc(policy, "infer", observation=observation(0))
    response = rpc(policy, "infer", observation=observation(0, force=True))
    with pytest.raises(ValueError, match="feedback"):
        rpc(policy, "infer", observation=observation(1, force=True))
    with pytest.raises(ValueError, match="step_id"):
        rpc(policy, "observe", step_id=99)
    with pytest.raises(ValueError, match="success"):
        rpc(policy, "end_episode", success=None)
    rpc(
        policy,
        "observe",
        step_id=response["step_id"],
        executed_actions=response["actions"][:1],
        next_observation=observation(1, force=True),
    )
    with pytest.raises(ValueError, match="partial"):
        rpc(policy, "infer")
    rpc(policy, "abort_episode")
    assert len(policy.service.replay) == 0
    assert not policy.service.journal.read_all()


def test_background_resume_budget_and_eval_no_training(tmp_path):
    policy, settings = make_policy(tmp_path, start=True)
    try:
        collect_episode(policy)
        with policy.service.condition:
            assert policy.service.condition.wait_for(lambda: policy.service.status()["ready_for_online"], timeout=10)
    finally:
        policy.close()
    restored = ChunkLearner.load(settings.learner.checkpoint_path)
    assert restored.steps == 2
    assert restored.runtime_state["warmup_ready_adds_total"] == 2
    eval_config = dataclasses.replace(
        settings, mode="eval", actor_checkpoint=settings.learner.checkpoint_path, collect=CollectConfig(enabled=False)
    )
    service = OnlineLearner(restored, None, eval_config)
    evaluated = OnlineRLTPolicy(FrozenVLA(), service, eval_config)
    weights = [p.detach().clone() for p in restored.actor.parameters()]
    assert rpc(evaluated, "begin_episode")["phase"] == "eval"
    assert rpc(evaluated, "infer", observation=observation(0))["source"] == "RL"
    rpc(evaluated, "end_episode", success=False)
    assert service.thread is None
    assert restored.steps == 2
    assert not service.train_once()
    assert all(torch.equal(old, current) for old, current in zip(weights, restored.actor.parameters(), strict=True))
    # Resume does not repeat warmup. New data earns a bounded online update budget.
    resumed = OnlineLearner(restored, policy.service.journal, settings, start=False)
    online = OnlineRLTPolicy(FrozenVLA(), resumed, settings)
    collect_episode(online, "online_episode")
    updates = 0
    while resumed.train_once():
        updates += 1
    assert updates == 2 * settings.learner.grad_updates_per_cycle


def test_disabled_collection_requires_existing_data_and_freeze(tmp_path):
    policy, settings = make_policy(tmp_path, freeze=True)
    disabled = dataclasses.replace(settings, collect=CollectConfig(enabled=False))
    with pytest.raises(ValueError, match="Collection disabled"):
        OnlineLearner(policy.service.learner, policy.service.journal, disabled, start=False)
    collect_episode(policy)
    service = OnlineLearner(policy.service.learner, policy.service.journal, disabled, start=False)
    assert service.train_once()
    assert service.train_once()
    assert not service.train_once()


def test_inference_yaml_and_force_training_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    normal = inference_config.load_config("openpi")
    rlt = inference_config.load_config("openpi_rlt")
    assert normal.mode == "vla"
    assert rlt.mode == "rlt"
    assert rlt.rlt.warmup == WarmupConfig(600, 20000)
    assert rlt.rlt.replay.directory == str(training_config.local_path("runs/openpi_rlt/replay"))
    trained = training_config.get_config("openpi_rlt")
    trained = dataclasses.replace(
        trained, rlt=dataclasses.replace(trained.rlt, force=ForceConfig(enabled=True, input="features", feature_dim=3))
    )
    training_config.save_config(trained, tmp_path)
    restored = training_config.get_config(tmp_path / "config.yaml")
    assert restored.rlt.force == trained.rlt.force
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump({"train_config": normal.train_config, "rlt": {"force": {"enabeld": True}}}))
    with pytest.raises(ValueError, match="Unknown rlt.force"):
        inference_config.load_config(str(path))


def test_actor_training_yaml_dispatch_and_force_mismatch(tmp_path):
    policy, settings = make_policy(tmp_path, force=True)
    collect_episode(policy, force=True)
    trained = training_config.get_config("openpi_rlt")
    trained = dataclasses.replace(
        trained,
        num_train_steps=2,
        model=dataclasses.replace(trained.model, action_horizon=4),
        rlt=dataclasses.replace(
            trained.rlt,
            stage="actor_critic",
            force=settings.force,
            replay=settings.replay,
            learner=settings.learner,
            actor=settings.actor,
            critic=settings.critic,
        ),
    )
    training_runner.train(trained)
    assert ChunkLearner.load(settings.learner.checkpoint_path).spec.grip_dim == 3
    with pytest.raises(ValueError, match="force"):
        training_runner.train(dataclasses.replace(trained, rlt=dataclasses.replace(trained.rlt, force=ForceConfig())))


def test_client_and_real_websocket_roundtrip(tmp_path):
    policy, _ = make_policy(tmp_path)
    server = WebsocketPolicyServer(policy, metadata=policy.metadata)

    async def exercise():
        async with serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as running:  # noqa: SLF001
            port = running.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as websocket:
                metadata = msgpack_numpy.unpackb(await websocket.recv())
                assert metadata["rlt_online_protocol"] == "openpi-rlt-online-v1"
                packer = msgpack_numpy.Packer()

                async def request(event, **values):
                    await websocket.send(packer.pack({"rlt_event": event, "episode_id": "socket", **values}))
                    return msgpack_numpy.unpackb(await websocket.recv())

                assert (await request("begin_episode"))["accepted"]
                response = await request("infer", observation=observation(0))
                await request(
                    "observe",
                    step_id=response["step_id"],
                    executed_actions=response["actions"][:1],
                    next_observation=observation(1),
                )
                assert (await request("end_episode", success=True))["success"]

    asyncio.run(exercise())
    assert policy.service.journal.read_all()[0].rewards[0] == 1

    class Transport:
        def get_server_metadata(self):
            return policy.metadata

        def infer(self, request, *, retry_on_disconnect):
            assert not retry_on_disconnect
            return policy.infer(request)

    client = RLTClientPolicy(transport=Transport())
    client.begin_episode("helper")
    result = client.infer(observation(0))
    client.observe(result["actions"][:1], observation(1))
    client.end_episode(success=False)
    assert client.episode_id is None
    assert json.loads((policy.service.journal.path / "manifest.json").read_text())["spec"]["grip_dim"] == 0


def test_inference_runner_online_resume_eval_and_force_rejection(tmp_path, monkeypatch):
    _, settings = make_policy(tmp_path)
    trained = training_config.get_config("openpi_rlt")
    trained = dataclasses.replace(trained, model=dataclasses.replace(trained.model, action_horizon=4))
    training_config.save_config(trained, tmp_path / "training")
    checkpoint, readout = tmp_path / "vla", tmp_path / "readout"
    checkpoint.mkdir()
    readout.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"frozen vla")
    (readout / "readout.safetensors").write_bytes(b"frozen readout")
    (readout / "manifest.json").write_text(
        json.dumps(
            {
                "architecture": "openpi-rlt-cross-attention-v1",
                "readout_config": {"use_mask": False},
                "image_only": True,
                "mask_protocol": "unmasked-image-patches-v1",
            }
        )
    )

    def create_frozen(*args, **kwargs):
        policy = FrozenVLA()
        policy.metadata["action_fps"] = 30.0
        return policy

    monkeypatch.setattr(server_runner.policy_config, "create_trained_policy", create_frozen)
    monkeypatch.setattr(server_runner, "attach_readout", lambda *args: None)
    value = inference_config.InferenceConfig(
        mode="rlt",
        train_config=str(tmp_path / "training/config.yaml"),
        checkpoint=str(checkpoint),
        readout=str(readout),
        device="cpu",
        rlt=settings,
    )
    path = tmp_path / "inference.yaml"

    def write_and_create(settings):
        path.write_text(yaml.safe_dump(dataclasses.asdict(dataclasses.replace(value, rlt=settings))))
        return server_runner.create_policy(server_runner.Args(config=str(path)))

    # Exercise actual layout extraction, YAML parser, runner, worker and saved model identity.
    policy = write_and_create(settings)
    try:
        collect_episode(policy)
        with policy.service.condition:
            assert policy.service.condition.wait_for(lambda: policy.service.status()["ready_for_online"], timeout=10)
    finally:
        policy.close()
    evaluated = write_and_create(
        dataclasses.replace(
            settings,
            mode="eval",
            actor_checkpoint=settings.learner.checkpoint_path,
            collect=CollectConfig(enabled=False),
        )
    )
    try:
        rpc(evaluated, "begin_episode", "eval")
        assert rpc(evaluated, "infer", "eval", observation=observation(0))["source"] == "RL"
        assert evaluated.service.thread is None
    finally:
        evaluated.close()
    with pytest.raises(ValueError, match="force/state/action"):
        write_and_create(
            dataclasses.replace(
                settings,
                mode="eval",
                actor_checkpoint=settings.learner.checkpoint_path,
                force=ForceConfig(enabled=True, input="features", feature_dim=3),
            )
        )
    (checkpoint / "model.safetensors").write_bytes(b"different frozen vla")
    with pytest.raises(ValueError, match="identity"):
        write_and_create(dataclasses.replace(settings, mode="eval", actor_checkpoint=settings.learner.checkpoint_path))
