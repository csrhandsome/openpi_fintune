"""Deployment switch validation and compiled shared-prefix inference checks."""

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.compile_inference import configure_torch_compile
from openpi.models_pytorch.rlinf import gemma
from openpi.models_pytorch.rlinf import model
from openpi.models_pytorch.rlinf import siglip
from openpi.serving import runner
from openpi.serving.config import InferenceConfig
from openpi.tests.models_pytorch.pi0_pytorch_test import TinyVision


@pytest.mark.parametrize("value", ["false", 1, None])
def test_compile_yaml_requires_boolean(value):
    with pytest.raises(ValueError, match="use_torch_compile"):
        InferenceConfig(use_torch_compile=value)


def test_compile_yaml_and_cli_override(tmp_path):
    path = tmp_path / "server.yaml"
    path.write_text("train_config: config/train/openpi.yaml\nuse_torch_compile: true\nuse_triton: false\n")
    settings = runner.load_settings(runner.Args(config=str(path)))
    assert settings.use_torch_compile is True
    assert settings.use_triton is False
    assert runner.load_settings(runner.Args(config=str(path), use_torch_compile=False)).use_torch_compile is False
    path.write_text("train_config: config/train/openpi.yaml\nuse_torch_compile: false\n")
    assert runner.load_settings(runner.Args(config=str(path), use_torch_compile=True)).use_torch_compile is True


@pytest.mark.parametrize("mode", ["vla", "rlt", "force"])
def test_server_passes_compile_to_policy(monkeypatch, mode):
    from openpi.rlt.force_rlt import serve as force_serving

    settings = InferenceConfig(mode=mode, use_torch_compile=True, checkpoint="/tmp/checkpoint", device="cuda")
    monkeypatch.setattr(runner, "load_settings", lambda _: settings)
    training = runner.training_config.get_config("openpi")
    monkeypatch.setattr(runner.training_config, "get_config", lambda _: training)
    monkeypatch.setattr(runner, "resolve_readout_directory", lambda _: None)
    monkeypatch.setattr(runner, "readout_vla_checkpoint", lambda checkpoint, _: checkpoint)
    monkeypatch.setattr(runner, "attach_readout", lambda *args: None)
    monkeypatch.setattr(runner, "create_rlt_policy", lambda policy, *args: policy)
    received = {}
    expected = object()

    def create(*args, **kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(runner.policy_config, "create_trained_policy", create)
    monkeypatch.setattr(force_serving, "create_policy_from_config", create)
    assert runner.create_policy(runner.Args()) is expected
    assert received["use_torch_compile"] is True
    assert received["use_triton"] is False


def test_compile_disabled_and_invalid_device():
    network = torch.nn.Linear(2, 2)
    configure_torch_compile(network, enabled=False)
    with pytest.raises(ValueError, match="eval"):
        configure_torch_compile(network, enabled=True)
    network.eval()
    with pytest.raises(ValueError, match="CUDA"):
        configure_torch_compile(network, enabled=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Inductor")
@pytest.mark.parametrize("pi05", [False, True])
@pytest.mark.parametrize("use_triton", [False, True])
def test_compiled_sampling_and_shared_prefix(monkeypatch, pi05, use_triton):
    from openpi.models_pytorch.triton_inference import configure_triton

    original_get_config = gemma.get_config
    monkeypatch.setattr(gemma, "get_config", lambda _: original_get_config("dummy"))
    monkeypatch.setattr(siglip, "SigLIPViT", TinyVision)
    config = SimpleNamespace(
        action_dim=32,
        action_horizon=2,
        max_token_len=4,
        pi05=pi05,
        dtype="float32",
        lora_rank=2,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    torch.manual_seed(12)
    network = pi0_pytorch.PI0Pytorch(config).cuda().eval()
    network.apply_lora_freeze()
    observation = model.Observation(
        images={key: torch.randn(1, 224, 224, 3, device="cuda") for key in model.IMAGE_KEYS},
        image_masks={key: torch.ones(1, dtype=torch.bool, device="cuda") for key in model.IMAGE_KEYS},
        state=torch.randn(1, 32, device="cuda"),
        tokenized_prompt=torch.ones(1, 4, dtype=torch.long, device="cuda"),
        tokenized_prompt_mask=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
    )
    observation.image_masks[model.IMAGE_KEYS[-1]] = torch.zeros(1, dtype=torch.bool, device="cuda")
    noise = torch.randn(1, 2, 32, device="cuda")
    keys = set(network.state_dict())
    configure_triton(network, enabled=use_triton)
    original_suffix = network.run_suffix
    original_prefix = network.build_prefix_cache
    with torch.no_grad():
        expected = network.sample_actions_with_prefix(observation, num_steps=2, noise=noise)
        configure_torch_compile(network, enabled=True)
        compiled_suffix = network.run_suffix
        assert compiled_suffix != original_suffix
        assert network.build_prefix_cache == original_prefix
        configure_torch_compile(network, enabled=True)
        assert network.run_suffix is compiled_suffix
        actual = network.sample_actions(observation, num_steps=2, noise=noise)
        shared = network.sample_actions_with_prefix(observation, num_steps=2, noise=noise)
        torch.testing.assert_close(actual, expected[0], atol=2e-4, rtol=2e-4)
        for output, reference in zip(shared, expected, strict=True):
            torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)
        # A second request must consume its own input and cache rather than stale values.
        changed = network.sample_actions(observation, num_steps=2, noise=noise + 0.5)
        assert not torch.allclose(actual, changed)
        assert set(network.state_dict()) == keys
        configure_torch_compile(network, enabled=False)
        assert network.run_suffix == original_suffix
        assert network.build_prefix_cache == original_prefix
        restored = network.sample_actions(observation, num_steps=2, noise=noise)
        torch.testing.assert_close(restored, expected[0], atol=0, rtol=0)
