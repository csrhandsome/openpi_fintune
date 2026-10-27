import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from openpi.training import config
from openpi.training import tracking
from openpi.training.tracking import init_tracking


@pytest.fixture(autouse=True)
def _isolate_tracking_paths(monkeypatch):
    for variable in ("SWANLAB_LOG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.delenv(variable, raising=False)


def test_mapping_without_backend_does_not_import_sdk(tmp_path, monkeypatch):
    load_sdk = Mock(side_effect=AssertionError("Disabled tracking must not import a tracker SDK"))
    monkeypatch.setattr(tracking.importlib, "import_module", load_sdk)
    assert init_tracking({}, tmp_path / "unused", resuming=False) is None
    assert not (tmp_path / "unused").exists()


def test_mapping_config_is_stored_without_train_config_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "LOG_ROOT", tmp_path / "data")
    settings = {
        "tracking_backend": "swanlab",
        "name": "force_encoder",
        "exp_name": "jepa",
        "project_name": "openpi",
        "training": {"max_steps": 4},
    }
    initialize = Mock(return_value=SimpleNamespace(id="jepa-run"))
    sdk = SimpleNamespace(init=initialize)
    monkeypatch.setattr(tracking.importlib, "import_module", Mock(return_value=sdk))
    assert init_tracking(settings, tmp_path, resuming=False) is sdk
    options = initialize.call_args.kwargs
    assert options["project"] == "openpi"
    assert options["experiment_name"] == "jepa"
    assert options["config"]["training"] == {"max_steps": 4}
    assert options["logdir"] == str(tmp_path / "data" / "swanlab" / "force_encoder" / "jepa")
    assert (tmp_path / "swanlab_id.txt").read_text() == "jepa-run"


def test_disabled_tracking_does_not_import_sdk(tmp_path, monkeypatch):
    load_sdk = Mock(side_effect=AssertionError("Disabled tracking must not import a tracker SDK"))
    monkeypatch.setattr(tracking.importlib, "import_module", load_sdk)
    settings = dataclasses.replace(config.get_config(), tracking_backend="none")
    assert init_tracking(settings, tmp_path / "unused", resuming=False) is None
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("backend", ["wandb", "swanlab"])
@pytest.mark.parametrize("existing_id", [None, "previous-run"])
def test_resume_uses_selected_sdk_and_its_own_id(tmp_path, monkeypatch, backend, existing_id):
    monkeypatch.setattr(tracking, "LOG_ROOT", tmp_path / "data")
    monkeypatch.delenv("SWANLAB_LOG_DIR", raising=False)
    settings = dataclasses.replace(config.get_config(), tracking_backend=backend)
    id_path = tmp_path / f"{backend}_id.txt"
    if existing_id:
        id_path.write_text(existing_id)
    other_backend = "wandb" if backend == "swanlab" else "swanlab"
    (tmp_path / f"{other_backend}_id.txt").write_text("other-backend-run")
    initialize = Mock(return_value=SimpleNamespace(id=existing_id or "new-run"))
    sdk = SimpleNamespace(init=initialize)
    load_sdk = Mock(return_value=sdk)
    monkeypatch.setattr(tracking.importlib, "import_module", load_sdk)
    assert init_tracking(settings, tmp_path, resuming=True) is sdk
    load_sdk.assert_called_once_with(backend)
    options = initialize.call_args.kwargs
    assert options["project"] == settings.project_name
    name_key, directory_key = ("name", "dir") if backend == "wandb" else ("experiment_name", "logdir")
    assert options[name_key] == settings.exp_name
    assert options["config"]["train"]["tracking_backend"] == backend
    assert options[directory_key] == str(tmp_path / "data" / backend / settings.name / settings.exp_name)
    assert id_path.read_text() == initialize.return_value.id
    assert (tmp_path / f"{other_backend}_id.txt").read_text() == "other-backend-run"
    if existing_id:
        assert options["id"] == existing_id
        assert options["resume"] == "must"
    else:
        assert "id" not in options
        assert "resume" not in options


@pytest.mark.parametrize("backend", ["wandb", "swanlab", "none"])
def test_tracking_yaml_cli_snapshot_roundtrip(tmp_path, backend):
    settings = config.cli(["--config", "openpi_rlt", "--tracking-backend", backend])
    assert settings.tracking_backend == backend
    config.save_config(settings, tmp_path)
    assert config.get_config(tmp_path / "config.yaml").tracking_backend == backend


def test_tracking_rejects_unknown_backend():
    with pytest.raises(ValueError, match="tracking_backend must be"):
        dataclasses.replace(config.get_config(), tracking_backend="unknown")


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("backend", ["wandb", "swanlab"])
@pytest.mark.parametrize("enabled", [True, False])
def test_legacy_checkpoint_logging_switch_is_migrated(tmp_path, nested, backend, enabled):
    values = config.to_dict(config.get_config())
    values["train"].pop("tracking_backend")
    target = values["train"] if nested else values
    target[f"{backend}_enabled"] = enabled
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values))
    settings = config.get_config(path)
    assert settings.tracking_backend == (backend if enabled else "none")
    assert f"{backend}_enabled" not in config.to_dict(settings)["train"]


@pytest.mark.parametrize("backend", ["wandb", "swanlab"])
def test_real_offline_sdk_records_metrics_and_images(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(tracking, "LOG_ROOT", tmp_path / "data")
    monkeypatch.delenv("SWANLAB_LOG_DIR", raising=False)
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("SWANLAB_SAVE_DIR", str(tmp_path / "credentials"))
    settings = dataclasses.replace(config.get_config(), tracking_backend=backend)
    sdk = init_tracking(settings, tmp_path, resuming=False)
    try:
        sdk.log(
            {"loss": 0.25, "camera_views": [sdk.Image(np.zeros((8, 16, 3), dtype=np.uint8))]},
            step=0,
        )
    finally:
        sdk.finish()
    assert list((tmp_path / "data" / backend).rglob("*.png"))
    if backend == "swanlab":
        assert (tmp_path / "swanlab_id.txt").read_text().strip()
    else:
        assert (tmp_path / "wandb_id.txt").read_text()
