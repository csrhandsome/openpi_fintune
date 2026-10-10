#!/usr/bin/env python3
"""Convert JAX Pi0/Pi0.5 weights to the RLinf PyTorch layout, with strict validation."""

import dataclasses
import json
import pathlib
from pathlib import Path
import shutil
from typing import Literal

import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp
from safetensors.torch import save_model
import torch
import tyro

from openpi.models_pytorch.jax_conversion import convert_llm
from openpi.models_pytorch.jax_conversion import convert_projections
from openpi.models_pytorch.jax_conversion import convert_siglip
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as training_config

_LORA_VARIANTS = {"gemma_2b_lora", "gemma_300m_lora"}
_DENSE_VARIANTS = {"dummy", "gemma_2b", "gemma_300m"}


def _unwrap(tree):
    if isinstance(tree, dict):
        if set(tree) == {"value"}:
            return _unwrap(tree["value"])
        return {key: _unwrap(value) for key, value in tree.items()}
    return np.asarray(tree, dtype=np.float32)


def restore_params(params_path: pathlib.Path | str, *, restore_type: type = np.ndarray, dtype=None, sharding=None):
    """Load an unstructured Orbax params pytree from an OpenPI JAX checkpoint."""
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    flat_params = flax.traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return flax.traverse_util.unflatten_dict(flat_params)


def _lora_scaling(variant: str, kind: str) -> float:
    if variant in _LORA_VARIANTS:
        return 1.0
    if variant in _DENSE_VARIANTS:
        raise ValueError(f"JAX {kind} LoRA requires a matching *_lora model variant")
    raise ValueError(f"Unknown variant: {variant}")


def load_jax_params(checkpoint_dir):
    checkpoint = Path(checkpoint_dir)
    params = restore_params(checkpoint / "params", restore_type=np.ndarray)
    while "PaliGemma" not in params:
        if "params" in params:
            params = params["params"]
        elif "vla" in params:
            params = params["vla"]
        else:
            raise ValueError("Checkpoint does not contain Pi0 PaliGemma parameters")
    return _unwrap(params)


def merge_jax_lora(params, config):
    """Merge native JAX adapters into dense weights before starting PEFT training."""
    layers = params["PaliGemma"]["llm"]["layers"]
    for expert, variant in enumerate((config.paligemma_variant, config.action_expert_variant)):
        suffix = "" if expert == 0 else "_1"
        if variant not in _LORA_VARIANTS | _DENSE_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for projection in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
            weights = layers["attn"][projection + suffix]
            if "lora_a" in weights or "lora_b" in weights:
                scaling = _lora_scaling(variant, "attention")
                weights["w"] = weights["w"] + (weights.pop("lora_a") @ weights.pop("lora_b")) * scaling
        mlp = layers["mlp" + suffix]
        for name in ("gating_einsum", "linear"):
            if name + "_lora_a" in mlp or name + "_lora_b" in mlp:
                scaling = _lora_scaling(variant, "MLP")
                mlp[name] = mlp[name] + (mlp.pop(name + "_lora_a") @ mlp.pop(name + "_lora_b")) * scaling
    return params


def convert_params(params, config):
    params = merge_jax_lora(_unwrap(params), config)
    state = {}
    for converted in (
        convert_siglip(params),
        convert_llm(params, pi05=config.pi05),
        convert_projections(params, pi05=config.pi05),
    ):
        state.update({key: value.contiguous() for key, value in converted.items()})
    return state


def validated_model(state, config, precision):
    # Meta construction validates the complete tensor layout without allocating
    # another random 2B backbone. Conversion exports dense weights; the runtime
    # initializes new PEFT adapters with a zero update if LoRA is requested.
    dense_config = dataclasses.replace(config, dtype=precision, is_lora=False)
    with torch.device("meta"):
        model = PI0Pytorch(dense_config)
    model.load_state_dict(state, strict=True, assign=True)
    model.to_bfloat16_for_selected_params(precision)
    return model


def convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config):
    checkpoint, output = Path(checkpoint_dir).resolve(), Path(output_path).resolve()
    if output == checkpoint or output in checkpoint.parents:
        raise ValueError("Conversion output must not overwrite the source checkpoint or its parent")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Conversion output already exists: {output}")
    model = validated_model(convert_params(load_jax_params(checkpoint), model_config), model_config, precision)
    output.mkdir(parents=True, exist_ok=True)
    save_model(model, output / "model.safetensors")
    assets = checkpoint / "assets"
    if not assets.is_dir() and (checkpoint.parent / "assets").is_dir():
        assets = checkpoint.parent / "assets"
    if assets.is_dir():
        shutil.copytree(assets, output / "assets")
    (output / "config.json").write_text(
        json.dumps({**dataclasses.asdict(model_config), "dtype": precision}, indent=2) + "\n"
    )
    print(f"Converted and validated all {len(model.state_dict())} tensors: {output / 'model.safetensors'}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    settings = training_config.get_config(config_name)
    model_config = settings.model
    if inspect_only:

        def inspect(tree, prefix=""):
            for name, value in tree.items():
                path = f"{prefix}/{name}".lstrip("/")
                if isinstance(value, dict):
                    inspect(value, path)
                else:
                    print(path, value.shape)

        inspect(load_jax_params(checkpoint_dir))
    elif output_path is None:
        raise ValueError("--output-path is required for conversion")
    else:
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    tyro.cli(main)
