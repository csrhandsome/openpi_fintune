import dataclasses
import json
import os
import pathlib
from unittest.mock import Mock

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.models import pi0_config
from openpi.tests.models_pytorch import checkpoint_alignment_test
from openpi.training import config as _config
from openpi.training import data_loader
from openpi.training import tracking
from openpi.training import vla as train
from openpi.training.validation import validate_dataset


@pytest.mark.parametrize("eval_enabled", [False, True])
def test_training_evaluates_denoised_actions_and_continues(tmp_path, monkeypatch, eval_enabled):
    checkpoint_alignment_test.tiny_backbone.__wrapped__(monkeypatch)
    monkeypatch.setattr(checkpoint_alignment_test.gemma, "PALIGEMMA_VOCAB_SIZE", 2048)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    settings = _config.TrainConfig(
        name="validation_flow",
        data=_config.FakeDataConfig(),
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
            max_token_len=4,
        ),
        requires_pytorch_weights=False,
        tracking_backend="none",
        batch_size=1,
        num_workers=0,
        checkpoint_base_dir=str(tmp_path),
        num_train_steps=2,
        log_interval=1,
        save_interval=2,
        eval_interval=1,
        eval_enabled=eval_enabled,
        eval_num_steps=10,
        pytorch_training_precision="float32",
    )
    dataset = data_loader.FakeDataset(settings.model, num_samples=20)
    dataset.episode_bounds = {index: (index, index + 1) for index in range(20)}
    create_dataset = Mock(return_value=dataset)
    monkeypatch.setattr(data_loader, "create_torch_dataset", create_dataset)
    evaluate = Mock(wraps=train._log_validation)  # noqa: SLF001
    monkeypatch.setattr(train, "_log_validation", evaluate)
    tracker = Mock()
    monkeypatch.setattr(train, "init_tracking", lambda *_args, **_kwargs: tracker)
    train.train_loop(settings)
    evaluations = [call for call in tracker.log.call_args_list if "eval_loss" in call.args[0]]
    assert [call.kwargs["step"] for call in evaluations] == ([1, 2] if eval_enabled else [])
    assert evaluate.call_count == (2 if eval_enabled else 0)
    # The disabled branch recreates the preview loader just as the original training path did.
    assert create_dataset.call_count == (1 if eval_enabled else 2)
    assert all(0 <= call.args[0]["eval_loss"] < float("inf") for call in evaluations)
    assert (settings.checkpoint_dir / "2" / "model.safetensors").is_file()
    tracker.finish.assert_called_once()


@pytest.mark.parametrize(
    ("lora_mode", "tracking_backend"),
    [(None, "none"), ("lora_only", "none"), ("openpi", "none"), (None, "swanlab"), (None, "wandb")],
)
def test_train(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, lora_mode, tracking_backend):
    checkpoint_alignment_test.tiny_backbone.__wrapped__(monkeypatch)
    monkeypatch.setattr(checkpoint_alignment_test.gemma, "PALIGEMMA_VOCAB_SIZE", 2048)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: False)
    if tracking_backend != "none":
        monkeypatch.setattr(tracking, "LOG_ROOT", tmp_path / "data")
        for variable in ("SWANLAB_LOG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setenv("SWANLAB_MODE", "offline")
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("SWANLAB_SAVE_DIR", str(tmp_path / "credentials"))
    config = dataclasses.replace(
        _config.get_config("openpi"),
        data=_config.FakeDataConfig(),
        requires_pytorch_weights=False,
        pytorch_weight_path=None,
        jax_weight_path=None,
        tracking_backend=tracking_backend,
        batch_size=1,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
            max_token_len=8,
            is_lora=lora_mode is not None,
            lora_mode=lora_mode or "lora_only",
            lora_rank=2,
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
        _config.get_config("openpi"),
        exp_name="missing-weights",
        data=_config.FakeDataConfig(),
        checkpoint_base_dir=str(tmp_path),
        pytorch_weight_path=str(tmp_path / "torch"),
        jax_weight_path=str(tmp_path / "missing-jax"),
        tracking_backend="none",
    )
    with pytest.raises(FileNotFoundError, match="JAX checkpoint not found"):
        train.train_loop(config)
    assert not config.checkpoint_dir.exists()


def test_missing_torch_weights_are_converted_once(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    from openpi.training.base_weights import ensure_pytorch_base_weights

    calls = []

    def convert(checkpoint_dir, output_dir, *, pi05, action_dim, action_horizon, max_token_len):
        del checkpoint_dir, action_dim, action_horizon, max_token_len
        calls.append(pi05)
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.safetensors").write_bytes(b"converted")

    monkeypatch.setattr("openpi.training.base_weights.convert_checkpoint", convert)
    jax_dir = tmp_path / "jax"
    (jax_dir / "params").mkdir(parents=True)
    config = dataclasses.replace(
        _config.get_config("openpi"),
        pytorch_weight_path=str(tmp_path / "torch"),
        jax_weight_path=str(jax_dir),
    )
    first = ensure_pytorch_base_weights(config, is_main=True)
    second = ensure_pytorch_base_weights(config, is_main=True)
    assert first == second
    assert pathlib.Path(first).read_bytes() == b"converted"
    assert calls == [False]


@pytest.mark.parametrize("config_name", ["openpi_rlt", "openpi"])
def test_default_ros_config_and_dataset_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, config_name, train_yaml
):
    default = _config.cli(args=[])
    default_yaml = train_yaml()
    assert (default.name, default.exp_name) == (default_yaml["name"], default_yaml["exp_name"])
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    config = dataclasses.replace(
        _config.get_config(config_name),
        data=dataclasses.replace(_config.get_config(config_name).data, repo_id="robot/session"),
    )
    yaml_config = train_yaml(config_name)
    assert config.data.action_space == yaml_config["data"]["action_space"]
    assert config.data.action_dim == yaml_config["data"]["action_dim"]
    assert config.model.lora_rank == yaml_config["model"]["lora_rank"]
    layout = config.data.layout
    target, dimension = layout.target, layout.motion_dim * layout.num_arms
    gripper_dimension = layout.gripper_dim * layout.num_arms
    dataset_dir = tmp_path / "robot/session"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": yaml_config["policy_metadata"]["action_fps"],
                "total_episodes": 1,
                "features": {
                    f"action.{target}": {"shape": [dimension]},
                    "action.gripper_position": {"shape": [gripper_dimension]},
                    "observation.gripper_position": {"shape": [gripper_dimension]},
                    f"observation.{target}": {"shape": [dimension]},
                },
            }
        )
    )
    sync_path = dataset_dir / "episode_000000.sync.json"
    sync_path.write_text(json.dumps({"action_space": "joint", "action_target_offset_frames": 1}))
    validate_dataset(config)

    sync_path.write_text(json.dumps({"action_semantics": "next_camera_observation_measured_state"}))
    validate_dataset(config)

    sync_path.write_text(json.dumps({"action_semantics": "current_measured_state"}))
    with pytest.raises(ValueError, match="Action timing"):
        validate_dataset(config)

    sync_path.write_text(json.dumps({"action_space": "joint", "action_target_offset_frames": 0}))
    with pytest.raises(ValueError, match="Action timing"):
        validate_dataset(config)
