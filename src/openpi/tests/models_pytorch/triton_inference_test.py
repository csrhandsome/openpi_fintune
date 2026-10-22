"""CUDA correctness checks for the optional inference-only Triton path."""

import pytest
import torch

from openpi.models_pytorch.rlinf import gemma
from openpi.serving import runner
from openpi.serving.config import InferenceConfig
from openpi.serving.runner import Args
from openpi.serving.runner import load_settings

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")


@pytest.mark.parametrize("value", ["false", 1, None])
def test_triton_yaml_requires_boolean(value):
    with pytest.raises(ValueError, match="use_triton"):
        InferenceConfig(use_triton=value)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("shape", [(2, 7, 1024), (2, 2048), (1, 3, 63)])
def test_rms_norm_matches_eager(dtype, adaptive, shape):
    from openpi.models_pytorch.triton_inference import configure_triton

    torch.manual_seed(7)
    module = gemma.RMSNorm(shape[-1], adaptive=adaptive).to(device="cuda", dtype=dtype).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.05)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    cond = torch.randn(shape[0], shape[-1], device="cuda", dtype=dtype) if adaptive else None
    with torch.no_grad():
        expected, expected_gate = module(x, cond)
        configure_triton(module, enabled=True)
        actual, actual_gate = module(x, cond)
    tolerance = 1e-6 if dtype == torch.float32 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    if adaptive:
        torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)
    else:
        assert actual_gate is None
    # Enabling inference kernels must not remove the autograd implementation.
    x.requires_grad_()
    module(x, cond)[0].float().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    module.train()
    with torch.no_grad():
        torch.testing.assert_close(module(x, cond)[0], expected, atol=0, rtol=0)


def test_disabled_triton_does_not_require_cuda():
    module = gemma.RMSNorm(64).eval()
    x = torch.randn(2, 3, 64, requires_grad=True)
    module(x)[0].sum().backward()
    assert x.grad is not None


def test_triton_yaml_and_cli_override(tmp_path):
    config = tmp_path / "server.yaml"
    config.write_text("train_config: config/train/openpi.yaml\nuse_triton: true\n")
    assert load_settings(Args(config=str(config))).use_triton is True
    assert load_settings(Args(config=str(config), use_triton=False)).use_triton is False
    config.write_text("train_config: config/train/openpi.yaml\nuse_triton: false\n")
    assert load_settings(Args(config=str(config), use_triton=True)).use_triton is True


@pytest.mark.parametrize("mode", ["vla", "force"])
def test_server_passes_triton_to_policy(monkeypatch, mode):
    from openpi.rlt.force_rlt import serve as force_serving

    settings = InferenceConfig(mode=mode, use_triton=True, checkpoint="/tmp/checkpoint", device="cuda")
    monkeypatch.setattr(runner, "load_settings", lambda _: settings)
    training = runner.training_config.get_config("openpi")
    monkeypatch.setattr(runner.training_config, "get_config", lambda _: training)
    received = {}
    expected = object()

    def create(*args, **kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(runner.policy_config, "create_trained_policy", create)
    monkeypatch.setattr(force_serving, "create_policy_from_config", create)
    assert runner.create_policy(Args()) is expected
    assert received["use_triton"] is True


@cuda
@pytest.mark.parametrize("adaptive", [False, True])
def test_gemma_prefix_and_cached_suffix(adaptive):
    from openpi.models_pytorch.triton_inference import configure_triton

    torch.manual_seed(21)
    config = gemma.get_config("dummy")
    module = gemma.Module([config, config], embed_dtype="float32", adarms=[False, adaptive]).cuda().eval()
    if adaptive:
        with torch.no_grad():
            for norm in module.modules():
                if isinstance(norm, gemma.RMSNorm) and norm.adaptive:
                    norm.ada_modulation.weight.normal_(std=0.01)
    prefix = torch.randn(2, 7, config.width, device="cuda")
    suffix = torch.randn(2, 3, config.width, device="cuda")
    cond = torch.randn(2, config.width, device="cuda") if adaptive else None
    prefix_positions = torch.arange(7, device="cuda")[None].expand(2, -1)
    suffix_positions = torch.arange(7, 10, device="cuda")[None].expand(2, -1)
    prefix_mask = torch.ones(2, 7, 7, device="cuda", dtype=torch.bool)
    suffix_mask = torch.ones(2, 3, 10, device="cuda", dtype=torch.bool)
    suffix_mask[:, :, 5:7] = False
    results = []
    keys_before = set(module.state_dict())
    with torch.no_grad():
        for enabled in (False, True):
            configure_triton(module, enabled=enabled)
            outputs, cache = module([prefix, None], positions=prefix_positions, mask=prefix_mask)
            actions, _ = module(
                [None, suffix],
                positions=suffix_positions,
                mask=suffix_mask,
                kv_cache=cache,
                adarms_cond=[None, cond],
            )
            results.append((outputs[0], actions[1]))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert set(module.state_dict()) == keys_before


@cuda
def test_enabled_triton_rejects_cpu():
    from openpi.models_pytorch.triton_inference import configure_triton

    with pytest.raises(ValueError, match="CUDA"):
        configure_triton(gemma.RMSNorm(64).eval(), enabled=True)
