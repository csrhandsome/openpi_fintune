"""Focused checks for the RLinf-backed PyTorch Pi0 training path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.rlinf import gemma, lora, model, siglip


class TinyVision(nn.Module):
    def __init__(self, *, num_classes, **_kwargs):
        super().__init__()
        self.proj = nn.Linear(3, num_classes)
        self.encoder = nn.Identity()
        self.encoder.gradient_checkpointing = False

    def forward(self, image):
        pooled = image.mean(dim=(1, 2)).to(self.proj.weight.dtype)
        return self.proj(pooled).unsqueeze(1), None


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sft_loss_and_sampling_with_lora(monkeypatch, pi05, dtype):
    original_get_config = gemma.get_config

    def tiny_config(variant):
        config = original_get_config("dummy")
        if "lora" in variant:
            config.lora_configs = {
                "attn": lora.LoRAConfig(rank=2, alpha=2),
                "ffn": lora.LoRAConfig(rank=2, alpha=2),
            }
        return config

    monkeypatch.setattr(gemma, "get_config", tiny_config)
    monkeypatch.setattr(siglip, "SigLIPViT", TinyVision)

    config = SimpleNamespace(
        action_dim=32,
        action_horizon=2,
        max_token_len=4,
        pi05=pi05,
        dtype=dtype,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    network = pi0_pytorch.PI0Pytorch(config)
    network.to_bfloat16_for_selected_params(dtype)
    observation = model.Observation(
        images={key: torch.randn(1, 224, 224, 3) for key in model.IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in model.IMAGE_KEYS},
        state=torch.randn(1, 32),
        tokenized_prompt=torch.ones(1, 4, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    observation.image_masks[model.IMAGE_KEYS[-1]] = torch.zeros(1, dtype=torch.bool)
    actions = torch.randn(1, 2, 32)

    network.apply_lora_freeze()
    assert not network.llm.layers[0].attn.q_proj[0].weight.requires_grad
    assert network.llm.layers[0].attn.q_proj[0].lora_a.requires_grad
    assert network.img.proj.weight.requires_grad
    network.gradient_checkpointing_enable()
    loss = network(observation, actions, train=True)
    assert loss.shape == actions.shape
    assert torch.isfinite(loss).all()
    assert not network.is_gradient_checkpointing_enabled()
    loss.mean().backward()
    assert network.llm.layers[0].attn.q_proj[0].lora_a.grad is not None

    network.eval()
    with torch.no_grad():
        sampled = network.sample_actions(observation, num_steps=2)
        fixed_noise = torch.randn_like(actions)
        original = network.sample_actions(observation, num_steps=2, noise=fixed_noise)
        shared, prefix, prefix_mask = network.sample_actions_with_prefix(
            observation, num_steps=2, noise=fixed_noise
        )
    assert sampled.shape == actions.shape
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(original, shared, rtol=0, atol=0)
    assert prefix.shape[:2] == prefix_mask.shape
    assert prefix_mask.sum() == 2  # only the two valid image cameras

    assert network.llm.layers[0].attn.q_proj[0].weight.dtype == (torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    assert network.llm.layers[0].pre_attention_norms[0].scale.dtype == torch.float32
    assert network.action_out_proj.weight.dtype == torch.float32
