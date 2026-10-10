from openpi.training import config
from openpi.training import lerobot_utils


def test_default_root_matches_collector_and_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    collector = config.REPO_ROOT.parent / "data_collect" / "data" / "dataset"
    assert lerobot_utils.resolve_dataset_root("openpi/franka_lerobot_test_100hz") == (
        collector / "franka_lerobot_test_100hz"
    )


def test_missing_dataset_uses_flat_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    assert lerobot_utils.resolve_dataset_root("robot/session") == tmp_path / "session"


def test_flat_dataset_takes_precedence_over_nested_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    for root in (tmp_path / "session", tmp_path / "robot" / "session"):
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}")
    assert lerobot_utils.resolve_dataset_root("robot/session") == tmp_path / "session"


def test_existing_nested_dataset_remains_accessible(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.delenv("OPENPI_DATASET_ROOT", raising=False)
    root = tmp_path / "robot" / "session"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    assert lerobot_utils.resolve_dataset_root("robot/session") == root


def test_explicit_dataset_root_overrides_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "unused"))
    monkeypatch.setenv("OPENPI_DATASET_ROOT", str(tmp_path / "selected"))
    assert lerobot_utils.resolve_dataset_root("robot/session") == tmp_path / "selected"
