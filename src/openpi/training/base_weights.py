"""Convert the official JAX Pi0 checkpoint once when Torch weights are absent."""

import logging
import os
import pathlib

import filelock
import torch.distributed as dist

from openpi.models_pytorch.jax_to_openpi import convert_checkpoint
from openpi.shared.download import maybe_download
from openpi.training.config import local_path

_OFFICIAL_JAX = {
    False: "gs://openpi-assets/checkpoints/pi0_base",
    True: "gs://openpi-assets/checkpoints/pi05_base",
}


def pytorch_weight_dir(config) -> pathlib.Path:
    """Directory that should contain the converted ``model.safetensors``."""
    configured = config.pytorch_weight_path
    if configured:
        path = pathlib.Path(configured)
        return path.parent if path.suffix == ".safetensors" else path
    name = "pi05_base" if config.model.pi05 else "pi0_base"
    return local_path(f"data/checkpoints/torch/{name}")


def jax_checkpoint_path(config) -> str:
    return config.jax_weight_path or os.environ.get("OPENPI_JAX_WEIGHT_PATH") or _OFFICIAL_JAX[bool(config.model.pi05)]


def ensure_pytorch_base_weights(config, *, is_main: bool) -> str:
    """Return the RLinf-layout safetensors path, converting the official JAX checkpoint once."""
    destination = pytorch_weight_dir(config)
    weights = destination / "model.safetensors"
    if weights.is_file():
        return str(weights)

    if not is_main:
        if dist.is_initialized():
            dist.barrier()
        if not weights.is_file():
            raise FileNotFoundError(f"PyTorch base weights were not converted: {weights}")
        return str(weights)

    source = jax_checkpoint_path(config)
    logging.info("PyTorch weights missing at %s; converting official JAX checkpoint %s once", weights, source)
    local_source = maybe_download(source) if "://" in source else pathlib.Path(source).expanduser()
    if not local_source.exists():
        raise FileNotFoundError(
            f"PyTorch weights not found at {weights}, and JAX checkpoint not found at {local_source}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(destination / ".jax_convert.lock"):
        if not weights.is_file():
            convert_checkpoint(
                local_source,
                destination,
                pi05=bool(config.model.pi05),
                action_dim=int(config.model.action_dim),
                action_horizon=int(config.model.action_horizon),
                max_token_len=int(config.model.max_token_len),
            )
    if dist.is_initialized():
        dist.barrier()
    if not weights.is_file():
        raise FileNotFoundError(f"JAX conversion did not write {weights}")
    logging.info("Converted official JAX checkpoint to %s", weights)
    return str(weights)
