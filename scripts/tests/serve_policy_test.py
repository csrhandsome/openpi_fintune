import pathlib

from scripts import serve_policy


def test_default_server_selects_latest_ros_checkpoint(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENPI_SERVER_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("OPENPI_SERVER_CONFIG", raising=False)
    monkeypatch.delenv("OPENPI_SERVER_EXP_NAME", raising=False)
    root = tmp_path / "checkpoints/pi0_base_ee_ros_30hz/ros_30hz"
    for step in (1, 10):
        checkpoint = root / str(step)
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"test")
    (root / "20").mkdir()

    checkpoint = serve_policy.latest_ros_checkpoint()
    assert serve_policy.Args().env is serve_policy.EnvMode.ROS
    assert checkpoint.config == "pi0_base_ee_ros_30hz"
    assert pathlib.Path(checkpoint.dir) == root / "10"
