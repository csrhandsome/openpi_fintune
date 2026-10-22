"""Optional forward-only Gemma kernels. Imported only when explicitly enabled.

Training and autograd always use the original PyTorch implementations. No
parameters are replaced, so checkpoint keys and dtypes remain unchanged.
"""

import logging

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(x, scale, shift, y, n: tl.constexpr, tokens: tl.constexpr, adaptive: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    x = tl.load(x + row * n + cols, cols < n, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / n + 1.0e-6)
    scale_row = row // tokens if adaptive else 0
    scale = tl.load(scale + scale_row * 3 * n + cols if adaptive else scale + cols, cols < n, 0).to(tl.float32)
    result = (x * inv_rms) * (1.0 + scale)
    if adaptive:
        shift = tl.load(shift + scale_row * 3 * n + cols, cols < n, 0).to(tl.float32)
        result = result + shift
    tl.store(y + row * n + cols, result, cols < n)


def rms_norm(x, scale, shift=None):
    """Gemma (1 + scale) RMSNorm, optionally with batched adaRMS shift."""
    x = x.contiguous()
    width = x.shape[-1]
    output = torch.empty_like(x)
    adaptive = shift is not None
    tokens = x.shape[-2] if adaptive and x.ndim == 3 else 1
    _rms_norm[(x.numel() // width,)](
        x,
        scale,
        shift if adaptive else scale,
        output,
        width,
        tokens,
        adaptive,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


def configure_triton(model, *, enabled: bool):
    """Toggle only this model; reject unsupported enabled configurations early."""
    from openpi.models_pytorch.rlinf.gemma import RMSNorm

    if enabled:
        parameter = next(model.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("use_triton=true requires a CUDA model")
        if model.training:
            raise ValueError("Triton acceleration requires model.eval()")
    count = 0
    for module in model.modules():
        if isinstance(module, RMSNorm):
            module.use_triton = enabled
            count += 1
    if enabled and not count:
        raise ValueError("Triton acceleration requires the RLinf Gemma backbone")
    logging.info("Triton inference enabled=%s for %d Gemma modules", enabled, count)
