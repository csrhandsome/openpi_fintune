"""Real tiny VLA/readout joint training, matched EMA exports and exact resume."""

import dataclasses
import json

import pytest
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.readout_checkpoint import readout_vla_checkpoint
from openpi.models_pytorch.rlinf.model import IMAGE_KEYS
from openpi.models_pytorch.rlinf.model import Observation
from openpi.tests.models_pytorch.checkpoint_alignment_test import tiny_backbone  # noqa: F401
from openpi.training import config as train_config
from openpi.training import rlt


def test_joint_ema_and_resume(tiny_backbone, tmp_path, monkeypatch):  # noqa: F811
    del tiny_backbone
    config = train_config.get_config("openpi_rlt")
    base_config = dataclasses.replace(
        config.model,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        is_lora=False,
        dtype="float32",
        action_horizon=2,
    )
    checkpoint = tmp_path / "vla"
    (checkpoint / "assets").mkdir(parents=True)
    torch.manual_seed(12)
    base = PI0Pytorch(base_config)
    save_file(base.state_dict(), checkpoint / "model.safetensors")
    model_config = dataclasses.replace(base_config, is_lora=True, lora_rank=2)
    config = dataclasses.replace(
        config,
        name="joint-fixture",
        model=model_config,
        pytorch_training_precision="float32",
        num_train_steps=2,
        save_interval=1,
        wandb_enabled=False,
        lr_schedule=dataclasses.replace(config.lr_schedule, warmup_steps=0),
        rlt=dataclasses.replace(
            config.rlt,
            vla_checkpoint=str(checkpoint),
            output_dir=str(tmp_path / "full"),
            embed_dim=8,
            alpha=1.0,
            num_heads=2,
            num_layers=1,
            max_prefix_len=16,
            device="cpu",
        ),
    )
    monkeypatch.setattr(rlt, "validate_dataset", lambda _: None)
    monkeypatch.setattr(
        train_config.LeRobotFrankaDataConfig,
        "create",
        lambda *args: type("Data", (), {"norm_stats": {"state": object()}})(),
    )
    observation = Observation(
        images={key: torch.randn(1, 224, 224, 3) for key in IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in IMAGE_KEYS},
        state=torch.randn(1, 32),
        tokenized_prompt=torch.ones(1, 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    actions = torch.randn(1, 2, 32)
    monkeypatch.setattr(rlt.data_loader, "create_data_loader", lambda *args, **kwargs: [(observation, actions)])
    rlt.train(config)
    full = torch.load(tmp_path / "full/2/training_state.pt", weights_only=True)
    assert any(name.endswith("lora_B.default.weight") and value.count_nonzero() for name, value in full["vla"].items())
    for name, original in base.state_dict().items():
        key = name if name in full["vla"] else name.rsplit(".", 1)[0] + ".base_layer." + name.rsplit(".", 1)[1]
        torch.testing.assert_close(full["vla"][key], original, rtol=0, atol=0)
    for module, filename in (("vla", "model.safetensors"), ("readout", "readout.safetensors")):
        exported = load_file(tmp_path / "full" / filename)
        for name, ema in full["ema"][module].items():
            torch.testing.assert_close(exported[name], ema, rtol=0, atol=0)
    manifest = json.loads((tmp_path / "full/manifest.json").read_text())
    assert manifest["joint_training"]
    assert not manifest["readout_config"]["use_mask"]
    assert manifest["stage1_settings"]["defaults"]["augment_images"]
    assert manifest["stage1_settings"]["optimizer"]["b2"] == 0.95
    assert readout_vla_checkpoint(checkpoint, tmp_path / "full") == tmp_path / "full/2"
    resumed = dataclasses.replace(
        config, num_train_steps=1, rlt=dataclasses.replace(config.rlt, output_dir=str(tmp_path / "resumed"))
    )
    rlt.train(resumed)
    rlt.train(dataclasses.replace(resumed, num_train_steps=2, resume=True))
    final = torch.load(tmp_path / "resumed/2/training_state.pt", weights_only=True)
    for module in ("vla", "readout"):
        for name, expected in full[module].items():
            torch.testing.assert_close(final[module][name], expected, rtol=0, atol=0)
            torch.testing.assert_close(final["ema"][module][name], full["ema"][module][name], rtol=0, atol=0)
    torch.testing.assert_close(final["rng"], full["rng"], rtol=0, atol=0)
    (tmp_path / "full/2/model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="matching VLA"):
        readout_vla_checkpoint(checkpoint, tmp_path / "full")
