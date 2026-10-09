import dataclasses
import json
import os
import pathlib

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import pi0_config
from openpi.training import config as _config

from ... import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=1,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=2, max_token_len=8
        ),
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        num_workers=0,
        pytorch_training_precision="float32",
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=2)
    train.main(config)


def test_finetune_config_requires_pytorch_checkpoint(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    config = dataclasses.replace(
        _config.get_config("pi0_libero"),
        exp_name="missing-weights",
        checkpoint_base_dir=str(tmp_path),
        wandb_enabled=False,
    )
    with pytest.raises(ValueError, match="requires PyTorch base weights"):
        train.train_loop(config)
    assert not config.checkpoint_dir.exists()


@pytest.mark.parametrize(
    ("config_name", "target", "dimension"),
    [
        ("pi0_rlt", "ee_pose", 6),
        ("pi0_base_ee_ros_30hz", "ee_pose", 6),
    ],
)
def test_default_ros_config_and_dataset_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, config_name, target, dimension
):
    default = _config.cli(args=[])
    assert (default.name, default.exp_name) == ("pi0_base_ee_ros_30hz", "ros_30hz")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    config = dataclasses.replace(
        _config.get_config(config_name),
        data=dataclasses.replace(_config.get_config(config_name).data, repo_id="robot/session"),
    )
    dataset_dir = tmp_path / "robot/session"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 1,
                "features": {
                    f"action.{target}": {"shape": [dimension]},
                    "action.gripper_position": {"shape": [1]},
                    "observation.gripper_position": {"shape": [1]},
                    f"observation.{target}": {"shape": [dimension]},
                },
            }
        )
    )
    sync_path = dataset_dir / "episode_000000.sync.json"
    sync_path.write_text(json.dumps({"action_space": "joint", "action_target_offset_frames": 1}))
    train.validate_ros_dataset(config)

    sync_path.write_text(json.dumps({"action_semantics": "next_camera_observation_measured_state"}))
    train.validate_ros_dataset(config)

    sync_path.write_text(json.dumps({"action_semantics": "current_measured_state"}))
    with pytest.raises(ValueError, match="Action timing"):
        train.validate_ros_dataset(config)

    sync_path.write_text(json.dumps({"action_space": "joint", "action_target_offset_frames": 0}))
    with pytest.raises(ValueError, match="Action timing"):
        train.validate_ros_dataset(config)
