"""Opt-in compilation of the shared Pi0 denoising step."""

import logging

import torch


def configure_torch_compile(model, *, enabled: bool):
    """Compile the inference denoising step without wrapping modules or changing checkpoint keys.

    Both ordinary VLA and shared-prefix RL-token inference call this method.
    Keep the prefix, preprocessing and Euler loop eager so ten steps reuse one
    suffix graph rather than compiling an unrolled solver. Keeping the prefix
    eager also limits BF16 numerical drift observed with vision compilation.
    Compilation is lazy; callers must warm up their actual deployment shapes before measuring latency.
    """
    originals = getattr(model, "_openpi_eager_inference", None)
    if not enabled:
        if originals is not None:
            for name, method in originals.items():
                setattr(model, name, method)
            delattr(model, "_openpi_eager_inference")
        return
    if model.training:
        raise ValueError("use_torch_compile=true requires an eval model")
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("use_torch_compile=true requires a CUDA model")
    if originals is not None:
        return
    names = ("run_suffix",)
    originals = {name: getattr(model, name) for name in names}
    compiled = {
        name: torch.compile(
            method,
            options={"emulate_precision_casts": True, "force_same_precision": True},
            fullgraph=True,
            dynamic=False,
        )
        for name, method in originals.items()
    }
    model._openpi_eager_inference = originals  # noqa: SLF001 -- reversible per-model compilation
    for name, method in compiled.items():
        setattr(model, name, method)
    logging.info("Enabled torch.compile for Pi0 denoising suffix; first inference compiles")
