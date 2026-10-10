"""Resolve complete Stage 1 snapshots and their matching VLA/readout pair."""

import json
from pathlib import Path


def resolve_readout_directory(directory):
    directory = Path(directory)
    candidates = (
        [path for path in directory.iterdir() if path.name.isdecimal() and (path / "training_state.pt").is_file()]
        if directory.is_dir()
        else []
    )
    return max(candidates, key=lambda path: int(path.name)) if candidates else directory


def readout_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "openpi-rlt-cross-attention-v1":
        raise ValueError("Readout uses the old architecture; retrain it with the aligned Stage 1 implementation")
    expected = "valid-image-patches-v1" if manifest["readout_config"].get("use_mask") else "unmasked-image-patches-v1"
    if not manifest["image_only"] or manifest["mask_protocol"] != expected:
        raise ValueError("Unsupported readout token selection/mask protocol")
    return manifest


def readout_vla_checkpoint(checkpoint, directory):
    directory = resolve_readout_directory(directory)
    manifest = readout_manifest(directory)
    if (directory / "model.safetensors").is_file():
        return directory
    if manifest.get("joint_training"):
        raise FileNotFoundError("Joint Stage 1 checkpoint is missing its matching VLA model.safetensors")
    return checkpoint
