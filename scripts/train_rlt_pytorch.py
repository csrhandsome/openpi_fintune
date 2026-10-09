"""Train only the PyTorch RL-token readout on frozen Pi0 image-prefix states.

Example: uv run python scripts/train_rlt_pytorch.py --config pi0_rlt
    --vla-checkpoint checkpoints/.../10000 --output checkpoints/readout --steps 5000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.models_pytorch.rlinf import model as pt_model
from openpi.training import config as train_config
from openpi.training import data_loader


def _device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.lr <= 0:
        parser.error("steps and lr must be positive")
    weight_path = args.vla_checkpoint / "model.safetensors"
    if not weight_path.is_file():
        parser.error(f"PyTorch VLA checkpoint missing: {weight_path}")
    config = train_config.get_config(args.config)
    if not isinstance(config.data, train_config.LeRobotFrankaEEDataConfig | train_config.LeRobotFrankaJointDataConfig):
        parser.error("The frozen VLA must use Franka external/wrist views without finger-camera inputs")
    action_space = "ee" if isinstance(config.data, train_config.LeRobotFrankaEEDataConfig) else "joint"
    # The readout sees the exact state normalization used by the frozen VLA.
    # A local training config may otherwise point at newer or unrelated stats.
    checkpoint_assets = args.vla_checkpoint / "assets"
    data = dataclasses.replace(
        config.data,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(checkpoint_assets)),
    )
    config = dataclasses.replace(config, data=data)
    if data.create(config.assets_dirs, config.model).norm_stats is None:
        parser.error(f"VLA checkpoint normalization stats missing under {checkpoint_assets}")
    vla = config.model.load_pytorch(config, str(weight_path)).to(args.device).eval()
    for parameter in vla.parameters():
        parameter.requires_grad = False
    if getattr(config.model, "pcd", False):
        parser.error("The image-only readout protocol does not accept point cloud VLA inputs")
    readout_cfg = RLTokenConfig(
        input_dim=vla.llm.configs[0].width,
        embed_dim=args.embed_dim,
        num_rl_tokens=args.num_tokens,
        num_layers=args.num_layers,
    )
    readout = RLTokenReadout(readout_cfg).to(args.device).train()
    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.lr)
    loader = data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    iterator = iter(loader)
    last_loss = None
    prefix_shape = None
    for step in range(args.steps):
        try:
            observation, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            observation, _ = next(iterator)
        observation = pt_model.Observation.from_observation_like(observation)
        observation = pt_model.Observation.from_dict(_device_tree(observation.to_dict(), args.device))
        with torch.no_grad():
            prefix, mask = vla.extract_prefix_embeddings(observation, image_only=True)
        prefix_shape = list(prefix.shape[1:])
        loss, tokens = readout.reconstruction_loss(prefix, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
        if (step + 1) % 100 == 0 or step == 0:
            print(f"step={step + 1} masked_reconstruction_mse={last_loss:.6f} token_std={float(tokens.std()):.6f}")
    args.output.mkdir(parents=True, exist_ok=True)
    readout_path = args.output / "readout.safetensors"
    save_file({key: value.detach().cpu().contiguous() for key, value in readout.state_dict().items()}, readout_path)
    manifest = {
        "format": "pytorch-rl-token-v1",
        "config": args.config,
        "readout_config": dataclasses.asdict(readout_cfg),
        "prefix_shape": prefix_shape,
        "image_only": True,
        "mask_protocol": "valid-image-patches-v1",
        "camera_protocol": "franka-droid-external-and-wrist-no-finger-v1",
        "action_horizon": vla.action_horizon,
        "model_action_dim": vla.action_dim,
        "action_space": action_space,
        "physical_action_dim": 7 if action_space == "ee" else 8,
        "final_masked_mse": last_loss,
    }
    if action_space == "joint":
        manifest["physical_joint_dim"] = 7
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
