import dataclasses
import pathlib

from openpi.serving import runner
from openpi.training import config


def test_default_server_selects_latest_torch_checkpoint(tmp_path: pathlib.Path):
    settings = dataclasses.replace(config.get_config("openpi"), checkpoint_base_dir=str(tmp_path))
    root = settings.checkpoint_dir
    for step in (1, 10):
        checkpoint = root / str(step)
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"test")
    (root / "20").mkdir()
    assert runner.latest_checkpoint(settings) == root / "10"
