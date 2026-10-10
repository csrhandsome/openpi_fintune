import dataclasses
import json

import numpy as np
import pytest
import yaml

from openpi import transforms
from openpi.training import config
from openpi.training import runner
from openpi.training.validation import validate_dataset


def _yaml(tmp_path, **changes):
    values = config.to_dict(config.get_config("openpi"))
    values.update(changes)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(values))
    return path


def _dataset(tmp_path, monkeypatch, motion_dim, grip_dim=1):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    root = tmp_path / "fixture"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 1,
                "features": {
                    f"{prefix}.{key}": {"shape": [dimension]}
                    for prefix in ("observation", "action")
                    for key, dimension in (
                        ("ee_pose", motion_dim),
                        ("joint_position", motion_dim),
                        ("gripper_position", grip_dim),
                    )
                },
            }
        )
    )
    (root / "episode_000000.sync.json").write_text(json.dumps({"action_target_offset_frames": 1}))
    return root


def test_yaml_cli_and_paths_are_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default = config.get_config()
    assert default.name == "openpi"
    assert default.model.is_lora is True
    assert default.model.lora_rank == 32
    assert config.gradient_accumulation(default, 1) == (1, default.batch_size)
    readout = config.get_config("openpi_rlt")
    assert readout.model.is_lora is True
    assert readout.rlt.embed_dim == 2048
    assert readout.rlt.alpha == 0.0
    assert default.assets_dirs == config.REPO_ROOT / "data/assets/openpi"
    selected = config.cli(
        [
            "--config",
            "config/train/openpi_rlt.yaml",
            "--batch-size",
            "8",
            "--model.action-horizon",
            "4",
            "--rlt.embed-dim",
            "16",
        ]
    )
    assert selected.name == "pi0_rlt"
    assert selected.mode == "rlt"
    assert selected.batch_size == 8
    assert selected.rlt.embed_dim == 16
    assert selected.policy_metadata["action_horizon"] == 4
    assert config.get_config("pi0_rlt").config_path == selected.config_path


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_szie": 8},
        {"batch_size": 0},
        {"mode": "jax"},
        {"model": {"type": "pi0_fast"}},
        {"model": {"freeze_filter": "anything"}},
        {"optimizer": {"type": "optax"}},
    ],
)
def test_reject_invalid_or_legacy_settings(tmp_path, changes):
    with pytest.raises(ValueError, match="Unknown|positive|must be|unsupported"):
        config.get_config(_yaml(tmp_path, **changes))


@pytest.mark.parametrize(
    ("space", "action_dim", "motion_dim", "grip_dim"),
    [
        ("ee", 7, 6, 1),
        ("ee_delta", 7, 6, 1),
        ("joint", 8, 7, 1),
        ("ee", 14, 12, 2),
        ("ee_delta", 14, 12, 2),
        ("joint", 16, 14, 2),
    ],
)
def test_action_dimensions_checked_before_training(tmp_path, monkeypatch, space, action_dim, motion_dim, grip_dim):
    _dataset(tmp_path, monkeypatch, motion_dim, grip_dim)
    data = config.LeRobotFrankaDataConfig(repo_id="robot/fixture", action_space=space, action_dim=action_dim)
    settings = dataclasses.replace(config.get_config(), data=data, checkpoint_base_dir=str(tmp_path / "checkpoints"))
    validate_dataset(settings)
    invalid = dataclasses.replace(settings, data=dataclasses.replace(data, action_dim=action_dim * 2))
    with pytest.raises(ValueError, match="Action dimension mismatch"):
        runner.train(invalid)
    assert not settings.checkpoint_dir.exists()


@pytest.mark.parametrize(("space", "dim"), [("ee", 7), ("ee_delta", 7), ("ee_delta", 14), ("joint", 8)])
def test_absolute_targets_and_delta_roundtrip(space, dim):
    settings = config.get_config()
    data = config.LeRobotFrankaDataConfig(repo_id="fake", action_space=space, action_dim=dim)
    pipeline = data.create(settings.assets_dirs, settings.model)
    layout = data.layout
    state_motion = np.linspace(0.1, 0.3, layout.motion_dim * layout.num_arms, dtype=np.float32)
    target_motion = np.stack([state_motion + 0.1, state_motion + 0.2])
    targets = layout.combine(target_motion, np.full((2, layout.num_arms), 0.25))
    sample = {
        "observation.exterior_image": np.zeros((4, 4, 3), np.uint8),
        "observation.wrist_image_left": np.zeros((4, 4, 3), np.uint8),
        f"observation.{layout.target}": state_motion,
        "observation.gripper_position": np.ones(layout.num_arms),
        f"action.{layout.target}": target_motion,
        "action.gripper_position": np.full((2, layout.num_arms), 0.25),
    }
    transformed = transforms.compose(pipeline.data_transforms.inputs)(sample)
    if space == "ee_delta":
        blocks = transformed["actions"].reshape(2, layout.num_arms, layout.arm_dim)
        np.testing.assert_allclose(blocks[0, :, :6], 0.1, atol=1e-6)
    else:
        np.testing.assert_allclose(transformed["actions"], targets)
    padded = transforms.PadStatesAndActions(32)(transformed)
    restored = transforms.compose(pipeline.data_transforms.outputs)(padded)
    np.testing.assert_allclose(restored["actions"], targets, atol=1e-6)


def test_effective_snapshot_roundtrip(tmp_path):
    settings = config.cli(["--batch-size", "8", "--exp-name", "snapshot"])
    config.save_config(settings, tmp_path)
    restored = config.get_config(tmp_path / "config.yaml")
    assert restored.batch_size == 8
    assert restored.exp_name == "snapshot"
    assert restored.data.action_space == "ee_delta"
    assert restored.checkpoint_dir == settings.checkpoint_dir
