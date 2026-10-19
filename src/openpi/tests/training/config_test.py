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


def _dataset(tmp_path, monkeypatch, motion_dim, grip_dim=1, *, fps):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    root = tmp_path / "fixture"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": fps,
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


def test_yaml_cli_and_paths_are_independent_of_cwd(tmp_path, monkeypatch, train_yaml):
    monkeypatch.chdir(tmp_path)
    default = config.get_config()
    yaml_config = train_yaml()
    assert default.name == yaml_config["name"]
    assert default.exp_name == yaml_config["exp_name"]
    for key in ("is_lora", "lora_rank", "lora_mode"):
        assert getattr(default.model, key) == yaml_config["model"][key]
    for key in (
        "num_gpus",
        "eval_enabled",
        "val_ratio",
        "split_seed",
        "eval_interval",
        "eval_num_steps",
        "micro_batch_size",
    ):
        assert getattr(default, key) == yaml_config["train"][key]
    assert default.data.action_space == yaml_config["data"]["action_space"]
    assert default.data.action_dim == yaml_config["data"]["action_dim"]
    assert default.policy_metadata["action_fps"] == yaml_config["policy_metadata"]["action_fps"]
    readout = config.get_config("openpi_rlt")
    readout_yaml = train_yaml("openpi_rlt")
    assert readout.model.is_lora == readout_yaml["model"]["is_lora"]
    assert readout.rlt.embed_dim == readout_yaml["rlt"]["embed_dim"]
    assert readout.rlt.alpha == readout_yaml["rlt"]["alpha"]
    assert readout.num_gpus == readout_yaml["train"]["num_gpus"]
    assert default.assets_dirs == (config.REPO_ROOT / yaml_config["train"]["assets_base_dir"] / default.name).resolve()
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
    assert selected.name == readout_yaml["name"]
    assert selected.mode == readout_yaml["mode"]
    assert selected.batch_size == 8
    assert selected.rlt.embed_dim == 16
    assert selected.policy_metadata["action_horizon"] == 4
    assert config.get_config("pi0_rlt").config_path == selected.config_path


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_szie": 8},
        {"batch_size": 0},
        {"val_ratio": 1},
        {"val_ratio": -0.1},
        {"eval_interval": 0},
        {"eval_enabled": "false"},
        {"eval_enabled": 1},
        {"mode": "jax"},
        {"model": {"type": "pi0_fast"}},
        {"model": {"freeze_filter": "anything"}},
        {"model": {"lora_mode": "anything"}},
        {"optimizer": {"type": "optax"}},
        {"data": {"type": "franka_ee"}},
        {"data": {"type": "franka_joint"}},
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
    default = config.get_config()
    _dataset(tmp_path, monkeypatch, motion_dim, grip_dim, fps=default.policy_metadata["action_fps"])
    data = config.LeRobotFrankaDataConfig(repo_id="robot/fixture", action_space=space, action_dim=action_dim)
    settings = dataclasses.replace(
        default,
        data=data,
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
    )
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
    settings = config.cli(["--batch-size", "8", "--exp-name", "snapshot", "--model.lora-mode", "openpi"])
    config.save_config(settings, tmp_path)
    restored = config.get_config(tmp_path / "config.yaml")
    assert restored.batch_size == 8
    assert restored.exp_name == "snapshot"
    assert restored.model.lora_mode == "openpi"
    assert restored.data.action_space == settings.data.action_space
    assert restored.val_ratio == settings.val_ratio
    assert restored.eval_enabled == settings.eval_enabled
    assert restored.split_seed == settings.split_seed
    assert restored.eval_interval == settings.eval_interval
    assert restored.eval_num_steps == settings.eval_num_steps
    assert restored.policy_metadata["action_fps"] == settings.policy_metadata["action_fps"]
    assert restored.checkpoint_dir == settings.checkpoint_dir


def test_eval_switch_yaml_cli_and_missing_field(tmp_path):
    values = config.to_dict(config.get_config())
    values["train"].pop("eval_enabled")
    path = tmp_path / "without_eval.yaml"
    path.write_text(yaml.safe_dump(values))
    assert config.get_config(path).eval_enabled is False
    enabled = config.cli(["--config", str(path), "--eval-enabled"])
    assert enabled.eval_enabled is True
    config.save_config(enabled, tmp_path / "enabled")
    assert config.get_config(tmp_path / "enabled/config.yaml").eval_enabled is True
    disabled = config.cli(["--config", str(tmp_path / "enabled/config.yaml"), "--no-eval-enabled"])
    assert disabled.eval_enabled is False


@pytest.mark.parametrize(("space", "dimension"), [("ee_delta", 7), ("joint", 8)])
def test_franka_yaml_retains_physical_dimensions(tmp_path, space, dimension):
    settings = config.get_config(
        _yaml(
            tmp_path,
            data={"type": "franka", "repo_id": "robot/fixture", "action_space": space, "action_dim": dimension},
        )
    )
    assert settings.data.action_dim == dimension
    assert settings.data.layout.action_dim == dimension
    config.save_config(settings, tmp_path / "snapshot")
    restored = config.get_config(tmp_path / "snapshot/config.yaml")
    assert type(restored.data) is type(settings.data)
    assert restored.data.layout == settings.data.layout

    with pytest.raises(ValueError, match="cannot be smaller than data.action_dim"):
        dataclasses.replace(settings, model=dataclasses.replace(settings.model, action_dim=dimension - 1))


@pytest.mark.parametrize(
    ("batch", "micro", "world", "expected"),
    [(5, None, 1, (5, 1)), (5, 1, 1, (1, 5)), (8, None, 2, (4, 1)), (8, 1, 2, (1, 4))],
)
def test_batch_splitting_with_and_without_accumulation(batch, micro, world, expected):
    settings = dataclasses.replace(config.get_config(), batch_size=batch, micro_batch_size=micro, num_gpus=world)
    assert config.gradient_accumulation(settings, world) == expected


@pytest.mark.parametrize("mode", ["lora_only", "openpi"])
def test_yaml_lora_mode_and_legacy_default(tmp_path, mode):
    model = config.to_dict(config.get_config())["model"]
    model["lora_mode"] = mode
    assert config.get_config(_yaml(tmp_path, model=model)).model.lora_mode == mode
    del model["lora_mode"]
    assert config.get_config(_yaml(tmp_path, model=model)).model.lora_mode == "lora_only"
