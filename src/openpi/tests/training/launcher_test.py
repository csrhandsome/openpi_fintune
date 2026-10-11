"""Automatic torchrun launch and manual-launch validation."""

import dataclasses
from pathlib import Path
import runpy
import sys

import pytest

from openpi.training import config as train_config


@pytest.mark.parametrize("name", ["openpi", "openpi_rlt"])
def test_automatic_two_process_launch(monkeypatch, name):
    script = Path(__file__).resolve().parents[4] / "scripts/train.py"
    namespace = runpy.run_path(str(script))
    settings = dataclasses.replace(train_config.get_config(name), batch_size=8, num_gpus=2)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    arguments = [str(script), "--config", f"config/train/{name}.yaml", "--batch-size", "8"]
    monkeypatch.setattr(sys, "argv", arguments)
    captured = []

    class RelaunchedError(Exception):
        pass

    def execvp(executable, argv):
        captured.append((executable, argv))
        raise RelaunchedError

    monkeypatch.setattr(namespace["os"], "execvp", execvp)
    with pytest.raises(RelaunchedError):
        namespace["_maybe_relaunch"](settings)
    assert captured == [
        (
            sys.executable,
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", *arguments],
        )
    ]
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    namespace["_maybe_relaunch"](settings)
    assert len(captured) == 1
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="WORLD_SIZE is 4"):
        namespace["_maybe_relaunch"](settings)


@pytest.mark.parametrize("stage", ["actor_critic", "force"])
def test_actor_stages_reject_multiple_processes(stage):
    script = Path(__file__).resolve().parents[4] / "scripts/train.py"
    namespace = runpy.run_path(str(script))
    config = train_config.get_config("openpi_rlt")
    settings = dataclasses.replace(config, num_gpus=2, rlt=dataclasses.replace(config.rlt, stage=stage))
    with pytest.raises(ValueError, match="use one process"):
        namespace["_maybe_relaunch"](settings)
