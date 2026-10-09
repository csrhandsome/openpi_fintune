"""Serve frozen PyTorch Pi0, RL-token readout, and B/C contact actor together."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_model
import torch

from openpi.force_rlt.finger_input import FingerRGBInput
from openpi.force_rlt.networks import ChunkActor
from openpi.force_rlt.networks import NetworkConfig
from openpi.force_rlt.tactile_encoder import SharedFingerEncoder
from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as train_config

# Engineering-machine client: /home/three/DataFromRoot/codes/grade_two/my_projects/franka_control_ros2/inference.py


class ForceRLTPolicy:
    """Run one VLA/readout inference and feed its reference to the contact actor."""

    def __init__(self, vla_policy, actor: ChunkActor, *, group: str, finger_input=None):
        self.vla_policy = vla_policy
        self.actor = actor.eval()
        self.group = group
        self.finger_input = finger_input

    def infer(self, obs: dict) -> dict:
        started = time.perf_counter()
        cfg = self.actor.cfg
        q = np.asarray(obs["observation.joint_position"], dtype=np.float32)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("observation.joint_position must be finite with shape (7,)")
        finger_started = time.perf_counter()
        if self.group == "C":
            mask = np.asarray(obs.get("finger_valid_mask", []), dtype=bool)
            if mask.shape != (4,) or not mask.all():
                raise ValueError("C requires a complete valid finger history")
            z_grip = (
                self.finger_input(obs) if self.finger_input is not None else np.asarray(obs["z_grip"], dtype=np.float32)
            )
            if z_grip.shape != (cfg.z_grip_dim,) or not np.isfinite(z_grip).all():
                raise ValueError(f"z_grip must be finite with shape ({cfg.z_grip_dim},)")
        else:
            z_grip = np.zeros(cfg.z_grip_dim, dtype=np.float32)

        finger_ms = (time.perf_counter() - finger_started) * 1000
        vla_obs = {key: value for key, value in obs.items() if not key.startswith("finger_") and key != "z_grip"}
        result = self.vla_policy.infer(vla_obs)
        z_rlt = np.asarray(result["z_rlt"], dtype=np.float32)
        ref_q = np.asarray(result["ref_q"], dtype=np.float32)
        if z_rlt.shape != (cfg.z_rlt_dim,) or not np.isfinite(z_rlt).all():
            raise ValueError("VLA RL-token shape or value differs from actor checkpoint")
        if ref_q.shape != (cfg.horizon, cfg.action_dim) or not np.isfinite(ref_q).all():
            raise ValueError("VLA reference shape or value differs from actor checkpoint")
        device = next(self.actor.parameters()).device
        inputs = [torch.tensor(value, device=device)[None] for value in (z_rlt, z_grip, q, ref_q)]
        actor_started = time.perf_counter()
        with torch.no_grad():
            proposal = self.actor.sample(*inputs, deterministic=True)[0].cpu().numpy()
        return {
            **result,
            "proposed_q": proposal,
            "group": self.group,
            "force_timing": {
                "finger_encode_ms": finger_ms,
                "actor_ms": (time.perf_counter() - actor_started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--group", choices=("B", "C"), required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument(
        "--finger-weights", type=Path, help="Local ResNet-18 weights; enables GPU RGB history input for C"
    )
    parser.add_argument(
        "--finger-roi",
        type=int,
        nargs=4,
        default=(0, 0, 224, 224),
        metavar=("X0", "Y0", "X1", "Y1"),
        help="ROI in the received, dataset-sized RGB frames",
    )
    parser.add_argument("--finger-max-skew-s", type=float, default=0.02)
    parser.add_argument("--finger-max-age-s", type=float, default=0.2)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest = json.loads((args.readout / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != "pytorch-rl-token-v1" or manifest["config"] != args.config:
        raise ValueError("Readout manifest protocol/config mismatch")
    if not manifest["image_only"] or manifest["mask_protocol"] != "valid-image-patches-v1":
        raise ValueError("Unsupported readout token selection/mask protocol")
    if not manifest.get("camera_protocol"):
        raise ValueError("Readout camera protocol is missing")
    config = train_config.get_config(args.config)
    if manifest.get("action_space", "joint") != "joint" or (config.policy_metadata or {}).get("action_space") == "ee":
        raise ValueError("The B/C contact actor requires joint references; EE VLA actions need an EE-to-joint adapter")
    saved = torch.load(args.actor_checkpoint, map_location="cpu", weights_only=True)
    if bool(saved["config"]["zero_grip"]) != (args.group == "B"):
        raise ValueError("Actor checkpoint group does not match B/C")
    network = NetworkConfig(**saved["config"]["network"])
    if network.action_dim != 7 or manifest["physical_joint_dim"] != 7:
        raise ValueError("Franka actor and readout must use seven joint targets")
    if network.horizon != manifest["action_horizon"]:
        raise ValueError("Actor horizon differs from frozen VLA")

    if (
        config.model.action_horizon != manifest["action_horizon"]
        or config.model.action_dim != manifest["model_action_dim"]
    ):
        raise ValueError("Readout VLA dimensions disagree with loaded config")
    readout = RLTokenReadout(RLTokenConfig(**manifest["readout_config"]))
    load_model(readout, args.readout / "readout.safetensors", strict=True)
    if network.z_rlt_dim != readout.config.num_rl_tokens * readout.config.embed_dim:
        raise ValueError("Actor RL-token dimension differs from readout")
    vla_policy = policy_config.create_trained_policy(config, args.vla_checkpoint, pytorch_device=args.device)
    vla_policy.attach_rl_readout(
        readout,
        metadata={
            "rlt_protocol": manifest["format"],
            "camera_protocol": manifest["camera_protocol"],
            "action_horizon": manifest["action_horizon"],
            "physical_joint_dim": manifest["physical_joint_dim"],
            "prefix_shape": manifest["prefix_shape"],
            "z_rlt_dim": network.z_rlt_dim,
        },
    )
    actor = ChunkActor(network).to(args.device)
    actor.load_state_dict(saved["actor"], strict=True)
    metadata = {
        **vla_policy.metadata,
        "rlt_actor_protocol": "pytorch-force-rlt-actor-v1",
        "group": args.group,
        "z_grip_dim": network.z_grip_dim,
        "prediction_horizon": network.horizon,
        "execution_horizon": network.executed_steps,
        "action_dim": network.action_dim,
    }
    finger_input = None
    if args.group == "C" and args.finger_weights is not None:
        if (
            not math.isfinite(args.finger_max_skew_s)
            or not math.isfinite(args.finger_max_age_s)
            or args.finger_max_skew_s <= 0
            or args.finger_max_age_s <= 0
        ):
            raise ValueError("Finger timing limits must be positive")
        encoder = SharedFingerEncoder(args.finger_weights, roi=tuple(args.finger_roi), device=args.device)
        finger_input = FingerRGBInput(encoder, max_skew_s=args.finger_max_skew_s, max_age_s=args.finger_max_age_s)
        if finger_input.feature_dim != network.z_grip_dim:
            raise ValueError("Finger encoder/history dimension differs from actor checkpoint")
        metadata.update(
            finger_input_protocol=finger_input.protocol,
            finger_history=4,
            finger_max_skew_s=finger_input.max_skew_s,
            finger_max_age_s=finger_input.max_age_s,
        )
    policy = ForceRLTPolicy(vla_policy, actor, group=args.group, finger_input=finger_input)
    WebsocketPolicyServer(policy, host="0.0.0.0", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
