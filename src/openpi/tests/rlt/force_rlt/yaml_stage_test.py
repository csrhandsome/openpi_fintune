"""Exercise contact training and serving through the shared YAML entry points."""

import dataclasses
import json

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
import yaml

from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.policies.policy import Policy
from openpi.rlt.force_rlt import serve as force_serving
from openpi.rlt.force_rlt.learner import ChunkLearner
from openpi.rlt.force_rlt.learner import LearnerConfig
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.rlt.force_rlt.replay import ReplayJournal
from openpi.serving import runner as server
from openpi.tests.rlt.force_rlt.force_rlt_test import record
from openpi.tests.serving.rlt_pytorch_smoke_test import FakeVLA
from openpi.training import config as training
from scripts import train


def settings(tmp_path, *, enabled=True, policy="actor"):
    base = training.get_config("openpi_rlt")
    return training.resolve_paths(
        dataclasses.replace(
            base,
            num_train_steps=1,
            save_interval=1,
            model=dataclasses.replace(base.model, action_horizon=4),
            data=dataclasses.replace(base.data, action_space="joint", action_dim=8),
            rlt=dataclasses.replace(
                base.rlt,
                stage="force",
                device="cpu",
                embed_dim=8,
                force=dataclasses.replace(base.rlt.force, enabled=enabled, input="features", feature_dim=12),
                force_stage=dataclasses.replace(
                    base.rlt.force_stage,
                    policy=policy,
                    replay_path=str(tmp_path / "replay.jsonl"),
                    checkpoint_path=str(tmp_path / "actor.pt"),
                    readout_dir=str(tmp_path / "readout"),
                    batch_size=2,
                    hidden_dim=16,
                    layers=1,
                ),
            ),
        )
    )


def write_yaml(config, path):
    path.write_text(yaml.safe_dump(training.to_dict(config)))
    return path


def network(config):
    stage = config.rlt.force_stage
    return NetworkConfig(
        z_rlt_dim=8,
        z_grip_dim=config.rlt.force.feature_dim,
        horizon=4,
        executed_steps=2,
        hidden_dim=stage.hidden_dim,
        layers=stage.layers,
        fixed_std=stage.fixed_std,
        max_reference_offset=stage.max_reference_offset_rad,
    )


def test_force_cli_overrides():
    config = training.cli(
        [
            "--config",
            "openpi_rlt",
            "--rlt.stage",
            "force",
            "--rlt.force.enabled",
            "--rlt.force-stage.policy",
            "vla",
        ]
    )
    assert config.rlt.stage == "force"
    assert config.rlt.force.enabled
    assert config.rlt.force_stage.policy == "vla"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source", ["RL", "HUMAN", "MIXED"])
def test_yaml_train_force_and_resume(tmp_path, monkeypatch, enabled, source):
    config = settings(tmp_path, enabled=enabled)
    transition = record(network(config), source="HUMAN" if source == "HUMAN" else "RL")
    if source == "MIXED":
        transition = dataclasses.replace(
            transition, source="MIXED", source_chunk=np.array(["RL", "HUMAN"]), intervention_flag=True
        )
    ReplayJournal(config.rlt.force_stage.replay_path, network(config)).append(transition)
    path = write_yaml(config, tmp_path / "train.yaml")
    monkeypatch.setattr("sys.argv", ["train.py", "--config", str(path)])
    train.main()
    saved = ChunkLearner.load(config.rlt.force_stage.checkpoint_path)
    assert saved.steps == 1
    assert saved.config.zero_grip is not enabled
    # Both current and next states respect the feature switch during training.
    for next_state in (False, True):
        grip = saved._state([record(network(config))], next_state=next_state)[1]  # noqa: SLF001
        torch.testing.assert_close(grip, torch.full_like(grip, float(enabled)))
    snapshot = training.get_config(tmp_path / "config.yaml")
    assert snapshot.rlt.force_stage == config.rlt.force_stage
    train.main(dataclasses.replace(config, resume=True))
    assert ChunkLearner.load(config.rlt.force_stage.checkpoint_path).steps == 2
    with pytest.raises(FileExistsError, match="output exists"):
        train.main(config)
    mismatch = dataclasses.replace(
        config,
        resume=True,
        rlt=dataclasses.replace(config.rlt, force=dataclasses.replace(config.rlt.force, enabled=not enabled)),
    )
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        train.main(mismatch)


def test_force_rejects_ee_vla_training_and_bad_replay(tmp_path):
    config = settings(tmp_path)
    ee = dataclasses.replace(config, data=dataclasses.replace(config.data, action_space="ee_delta", action_dim=7))
    with pytest.raises(ValueError, match="joint layout"):
        train.main(training.resolve_paths(ee))
    with pytest.raises(ValueError, match="no actor to train"):
        train.main(settings(tmp_path, policy="vla"))
    ReplayJournal(config.rlt.force_stage.replay_path, network(config)).append(record(network(config), source="HUMAN"))
    wrong = dataclasses.replace(
        config, rlt=dataclasses.replace(config.rlt, force=dataclasses.replace(config.rlt.force, feature_dim=24))
    )
    with pytest.raises(ValueError, match="shape"):
        train.main(wrong)


def readout_fixture(config):
    from pathlib import Path

    path = Path(config.rlt.force_stage.readout_dir)
    path.mkdir()
    readout = RLTokenReadout(
        RLTokenConfig(input_dim=8, embed_dim=8, num_rl_tokens=1, num_layers=1, num_heads=8, use_mask=False)
    )
    save_file(readout.state_dict(), path / "readout.safetensors")
    (path / "model.safetensors").write_bytes(b"fake VLA")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "pytorch-rl-token-v1",
                "architecture": "openpi-rlt-cross-attention-v1",
                "config": config.name,
                "image_only": True,
                "mask_protocol": "unmasked-image-patches-v1",
                "readout_config": dataclasses.asdict(readout.config),
                "camera_protocol": "fixture-camera",
                "action_space": "joint",
                "action_representation": "joint",
                "physical_action_dim": 8,
                "physical_joint_dim": 7,
                "action_horizon": 4,
                "model_action_dim": 32,
                "prefix_shape": [3, 8],
            }
        )
    )


@pytest.mark.parametrize(
    ("enabled", "policy_mode", "rgb"),
    [
        (False, "actor", False),
        (True, "actor", False),
        (True, "actor", True),
        (True, "vla", False),
    ],
)
def test_yaml_force_service(tmp_path, monkeypatch, enabled, policy_mode, rgb):
    config = settings(tmp_path, enabled=enabled, policy=policy_mode)
    if rgb:
        config = dataclasses.replace(
            config,
            rlt=dataclasses.replace(
                config.rlt,
                force=dataclasses.replace(
                    config.rlt.force, input="rgb", feature_dim=12288, weights="/tmp/finger-fixture.pt"
                ),
            ),
        )

        class Encoder:
            feature_dim = 512

            def encode_pairs(self, left, right):
                return np.ones((len(left), 2, 512), np.float32)

        monkeypatch.setattr(force_serving, "SharedFingerEncoder", lambda *a, **kw: Encoder())
    readout_fixture(config)
    if policy_mode == "actor":
        ChunkLearner(LearnerConfig(network(config), zero_grip=not enabled)).save(config.rlt.force_stage.checkpoint_path)

    def fake_policy(*args, **kwargs):
        return Policy(
            FakeVLA(),
            transforms=(),
            output_transforms=(lambda x: {"actions": x["actions"][:, :8]},),
            pytorch_device="cpu",
            is_pytorch=True,
            metadata=config.policy_metadata,
        )

    monkeypatch.setattr(server.policy_config, "create_trained_policy", fake_policy)
    train_path = write_yaml(config, tmp_path / "train.yaml")
    policy = server.create_policy(server.Args(config=str(train_path)))
    assert policy.metadata["force_policy"] == policy_mode
    assert policy.metadata["force_enabled"] == (enabled and policy_mode == "actor")
    assert "group" not in policy.metadata
    request = {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        "state": np.zeros(8, np.float32),
        "observation.joint_position": np.zeros(7, np.float32),
        "monotonic_ts": np.array(10.0),
        "O_T_TCP": np.eye(4, dtype=np.float32),
    }
    if enabled and policy_mode == "actor":
        request.update(z_grip=np.ones(12, np.float32), finger_valid_mask=np.ones(4, bool))
    if rgb:
        request.pop("z_grip")
        request.update(
            {
                "observation.gripper_image_left": np.ones((4, 2, 2, 3), np.uint8),
                "observation.gripper_image_right": np.ones((4, 2, 2, 3), np.uint8),
                "finger_timestamps": np.repeat(np.array([9.9, 9.93, 9.96, 9.99])[:, None], 2, axis=1),
                "finger_baseline_left": np.zeros((2, 2, 3), np.uint8),
                "finger_baseline_right": np.zeros((2, 2, 3), np.uint8),
                "finger_session_id": "yaml-rgb",
            }
        )
    response = policy.infer(request)
    assert response["proposed_q"].shape == (4, 7)
    assert "group" not in response
    if policy_mode == "vla":
        np.testing.assert_array_equal(response["proposed_q"], response["ref_q"])
    elif not enabled:
        changed = policy.infer({**request, "z_grip": np.full(12, 100, np.float32)})
        np.testing.assert_array_equal(response["proposed_q"], changed["proposed_q"])
    else:
        with pytest.raises(ValueError, match="valid finger history"):
            policy.infer({**request, "finger_valid_mask": [False] * 4})
        missing_anchor = dict(request)
        missing_anchor.pop("O_T_TCP")
        with pytest.raises(ValueError, match="measured O_T_TCP"):
            policy.infer(missing_anchor)

    inference_path = tmp_path / "inference.yaml"
    inference_path.write_text(yaml.safe_dump({"mode": "force", "train_config": str(train_path)}))
    deployed = server.create_policy(server.Args(config=str(inference_path)))
    assert deployed.metadata["force_enabled"] == policy.metadata["force_enabled"]
    if policy_mode == "actor":
        wrong = dataclasses.replace(
            config,
            rlt=dataclasses.replace(config.rlt, force=dataclasses.replace(config.rlt.force, enabled=not enabled)),
        )
        write_yaml(wrong, train_path)
        with pytest.raises(ValueError, match="checkpoint differs from force.enabled"):
            server.create_policy(server.Args(config=str(train_path)))
