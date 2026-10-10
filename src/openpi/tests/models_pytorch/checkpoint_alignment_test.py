"""Backbone completeness, JAX conversion and RLinf precision/adapter round trips."""

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.rlinf import gemma
from openpi.models_pytorch.rlinf import model as torch_model
from openpi.models_pytorch.rlinf import siglip

spec = importlib.util.spec_from_file_location(
    "checkpoint_converter", Path(__file__).resolve().parents[4] / "examples/convert_jax_model_to_pytorch.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


@pytest.fixture
def tiny_backbone(monkeypatch):
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8),
    )
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda _: {"width": 8, "depth": 1, "mlp_dim": 16, "num_heads": 2, "patch_size": (112, 112)},
    )


def jax_parameters(network):
    """Recreate JAX tensor shapes from a complete, independently initialized network."""
    state = {name: value.detach().cpu().numpy().copy() for name, value in network.state_dict().items()}
    img = {
        "embedding": {"kernel": state["img.stem.weight"].transpose(2, 3, 1, 0), "bias": state["img.stem.bias"]},
        "pos_embedding": state["img.pos_embedding"],
        "head": {"kernel": state["img.head.weight"].T, "bias": state["img.head.bias"]},
    }
    prefix = "img.encoder.layers.0"
    eb = {}
    for index, name in enumerate(("norm1", "norm2")):
        eb[f"LayerNorm_{index}"] = {
            "scale": state[f"{prefix}.{name}.weight"][None],
            "bias": state[f"{prefix}.{name}.bias"][None],
        }
    eb["MlpBlock_0"] = {
        f"Dense_{i}": {
            "kernel": state[f"{prefix}.mlp.fc{i + 1}.weight"].T[None],
            "bias": state[f"{prefix}.mlp.fc{i + 1}.bias"][None],
        }
        for i in range(2)
    }
    attn = {}
    for i, name in enumerate(("query", "key", "value")):
        attn[name] = {
            "kernel": state[f"{prefix}.attn.in_proj_weight"][i * 8 : (i + 1) * 8].T.reshape(1, 8, 2, 4),
            "bias": state[f"{prefix}.attn.in_proj_bias"][i * 8 : (i + 1) * 8].reshape(1, 2, 4),
        }
    attn["out"] = {
        "kernel": state[f"{prefix}.attn.out_proj.weight"].T.reshape(1, 2, 4, 8),
        "bias": state[f"{prefix}.attn.out_proj.bias"][None],
    }
    eb["MultiHeadDotProductAttention_0"] = attn
    img["Transformer"] = {
        "encoderblock": eb,
        "encoder_norm": {"scale": state["img.encoder.norm.weight"], "bias": state["img.encoder.norm.bias"]},
    }
    layers = {"attn": {}}
    llm = {"layers": layers, "embedder": {"input_embedding": state["llm.embedder.embedding.weight"]}}
    for expert in range(2):
        suffix = "" if expert == 0 else "_1"

        def stack(name):
            return np.stack([state[f"llm.layers.{i}.{name}"] for i in range(2)])

        layers["attn"]["q_einsum" + suffix] = {
            "w": stack(f"attn.q_proj.{expert}.weight").reshape(2, 2, 8, 16).transpose(0, 1, 3, 2)
        }
        layers["attn"]["kv_einsum" + suffix] = {
            "w": np.stack(
                [
                    stack(f"attn.{name}_proj.{expert}.weight").reshape(2, 1, 8, 16).transpose(0, 1, 3, 2)
                    for name in ("k", "v")
                ],
                axis=1,
            )
        }
        layers["attn"]["attn_vec_einsum" + suffix] = {
            "w": stack(f"attn.o_proj.{expert}.weight").transpose(0, 2, 1).reshape(2, 2, 8, 16)
        }
        layers["mlp" + suffix] = {
            "gating_einsum": stack(f"mlps.{expert}.w_gating"),
            "linear": stack(f"mlps.{expert}.w_linear"),
        }
        for name, torch_name in (("pre_attention_norm", "pre_attention_norms"), ("pre_ffw_norm", "pre_ffw_norms")):
            layers[name + suffix] = (
                {
                    "Dense_0": {
                        "kernel": stack(f"{torch_name}.{expert}.ada_modulation.weight").transpose(0, 2, 1),
                        "bias": stack(f"{torch_name}.{expert}.ada_modulation.bias"),
                    }
                }
                if network.pi05 and expert
                else {"scale": stack(f"{torch_name}.{expert}.scale")}
            )
        final = f"llm.final_norms.{expert}"
        llm["final_norm" + suffix] = (
            {
                "Dense_0": {
                    "kernel": state[final + ".ada_modulation.weight"].T,
                    "bias": state[final + ".ada_modulation.bias"],
                }
            }
            if network.pi05 and expert
            else {"scale": state[final + ".scale"]}
        )
    result = {"PaliGemma": {"img": img, "llm": llm}}
    for name in (
        ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
        if network.pi05
        else ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
    ):
        result[name] = {"kernel": state[name + ".weight"].T, "bias": state[name + ".bias"]}
    return result


@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_complete_jax_conversion_and_runtime_load(tiny_backbone, tmp_path, monkeypatch, pi05, precision):
    del tiny_backbone
    config = pi0_config.Pi0Config(
        pi05=pi05, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32", action_horizon=2
    )
    original = pi0_pytorch.PI0Pytorch(config)
    params = jax_parameters(original)
    converted = converter.convert_params(params, config)
    assert converted.keys() == original.state_dict().keys()
    for name, expected in original.state_dict().items():
        torch.testing.assert_close(converted[name], expected, rtol=0, atol=0)
    monkeypatch.setattr(converter, "load_jax_params", lambda _: params)
    checkpoint = tmp_path / "jax"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "assets/stats.json").write_text("{}")
    output = tmp_path / "torch"
    converter.convert_pi0_checkpoint(checkpoint, precision, output, config)
    settings = type("Settings", (), {"model": config, "pytorch_training_precision": precision})()
    restored = config.load_pytorch(settings, output / "model.safetensors")
    original.to_bfloat16_for_selected_params(precision)
    assert restored.embed_dtype == (torch.bfloat16 if precision == "bfloat16" else torch.float32)
    for name, expected in original.state_dict().items():
        actual = restored.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (output / "assets/stats.json").is_file()
    observation = torch_model.Observation(
        images={key: torch.randn(1, 224, 224, 3) for key in torch_model.IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool) for key in torch_model.IMAGE_KEYS},
        state=torch.randn(1, 32),
        tokenized_prompt=torch.ones(1, 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    loss, prefix, _ = restored.compute_loss_with_prefix(observation, torch.randn(1, 2, 32), train=False)
    assert torch.isfinite(loss).all()
    assert not prefix.requires_grad
    loss.mean().backward()
    with torch.no_grad():
        assert torch.isfinite(restored.sample_actions(observation, num_steps=1)).all()
    converted.pop("llm.embedder.embedding.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        converter.validated_model(converted, config, precision)
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_weights(converted)


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_peft_initial_output_and_checkpoint_roundtrip(tiny_backbone, tmp_path, precision):
    del tiny_backbone
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype=precision,
        action_horizon=2,
        is_lora=True,
        lora_rank=2,
    )
    model = pi0_pytorch.PI0Pytorch(config)
    model.to_bfloat16_for_selected_params(precision)
    inputs = torch.randn(2, 32)
    expected = model.action_in_proj(inputs).detach()
    model.apply_lora_freeze()
    torch.testing.assert_close(model.action_in_proj(inputs), expected, rtol=0, atol=0)
    assert all("lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    with torch.no_grad():
        model.action_in_proj.lora_B["default"].weight.normal_()
    path = tmp_path / "model.safetensors"
    save_file(model.state_dict(), path)
    settings = type("Settings", (), {"model": config, "pytorch_training_precision": precision})()
    restored = config.load_pytorch(settings, path)
    torch.testing.assert_close(restored.action_in_proj(inputs), model.action_in_proj(inputs), rtol=0, atol=0)
    assert load_file(path).keys() == restored.state_dict().keys()
    base = dataclasses.replace(config, is_lora=False)
    with pytest.raises(ValueError, match="PEFT checkpoint requires"):
        base.load_pytorch(type("Settings", (), {"model": base, "pytorch_training_precision": precision})(), path)


def test_jax_native_lora_is_merged_before_conversion(tiny_backbone):
    del tiny_backbone
    config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", is_lora=False, dtype="float32"
    )
    params = jax_parameters(pi0_pytorch.PI0Pytorch(config))
    expected = converter.convert_params(params, config)
    layers = params["PaliGemma"]["llm"]["layers"]
    rng = np.random.default_rng(13)
    for suffix in ("", "_1"):
        for name in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
            entry = layers["attn"][name + suffix]
            shape = entry["w"].shape
            entry["lora_a"] = rng.normal(size=(*shape[:-1], 2)).astype(np.float32)
            entry["lora_b"] = rng.normal(size=(*shape[:-2], 2, shape[-1])).astype(np.float32)
        for name in ("gating_einsum", "linear"):
            entry = layers["mlp" + suffix]
            shape = entry[name].shape
            entry[name + "_lora_a"] = rng.normal(size=(*shape[:-1], 2)).astype(np.float32)
            entry[name + "_lora_b"] = rng.normal(size=(*shape[:-2], 2, shape[-1])).astype(np.float32)
    converted = converter.convert_params(params, config)
    entry = layers["attn"]["q_einsum"]
    merged = entry["w"] + entry["lora_a"] @ entry["lora_b"]
    torch.testing.assert_close(
        converted["llm.layers.0.attn.q_proj.0.weight"],
        torch.from_numpy(merged[0].transpose(0, 2, 1).reshape(-1, 16)),
        rtol=0,
        atol=0,
    )
    for name in ("llm.layers.0.attn.o_proj.1.weight", "llm.layers.0.mlps.0.w_gating", "llm.layers.0.mlps.1.w_linear"):
        assert not torch.equal(converted[name], expected[name])
    assert "lora_a" in entry  # conversion leaves the original checkpoint tree intact
