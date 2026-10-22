"""Serve a frozen VLA reference or contact actor selected by YAML settings."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
from safetensors.torch import load_model
import torch

from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.policies import policy_config
from openpi.rlt.force_rlt.config import validate_joint_config
from openpi.rlt.force_rlt.finger_input import FingerRGBInput
from openpi.rlt.force_rlt.networks import ChunkActor
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.rlt.force_rlt.tactile_encoder import SharedFingerEncoder
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as train_config

# Engineering-machine client: /home/three/DataFromRoot/codes/grade_two/my_projects/franka_control_ros2/inference.py


class ForceRLTPolicy:
    """Run one VLA/readout inference and feed its reference to the contact actor."""

    def __init__(
        self,
        vla_policy,
        actor: ChunkActor | None,
        *,
        group: str | None = None,
        force_enabled: bool | None = None,
        finger_input=None,
        metadata=None,
    ):
        self.vla_policy = vla_policy
        self.actor = actor.eval() if actor is not None else None
        self.group = group
        self.force_enabled = group == "C" if force_enabled is None else force_enabled
        self.finger_input = finger_input
        self._metadata = metadata if metadata is not None else vla_policy.metadata

    @property
    def metadata(self):
        return self._metadata

    def infer(self, obs: dict) -> dict:
        started = time.perf_counter()
        q = np.asarray(obs["observation.joint_position"], dtype=np.float32)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("observation.joint_position must be finite with shape (7,)")
        if self.actor is None:
            result = self.vla_policy.infer(_vla_observation(obs))
            return {
                **result,
                "proposed_q": np.asarray(result["ref_q"]).copy(),
                "force_policy": "vla",
                "force_enabled": False,
            }
        cfg = self.actor.cfg
        finger_started = time.perf_counter()
        if self.force_enabled:
            mask = np.asarray(obs.get("finger_valid_mask", []), dtype=bool)
            if mask.shape != (4,) or not mask.all():
                raise ValueError("Force actor requires a complete valid finger history")
            z_grip = (
                self.finger_input(obs) if self.finger_input is not None else np.asarray(obs["z_grip"], dtype=np.float32)
            )
            if z_grip.shape != (cfg.z_grip_dim,) or not np.isfinite(z_grip).all():
                raise ValueError(f"z_grip must be finite with shape ({cfg.z_grip_dim},)")
        else:
            z_grip = np.zeros(cfg.z_grip_dim, dtype=np.float32)

        finger_ms = (time.perf_counter() - finger_started) * 1000
        vla_obs = _vla_observation(obs)
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
            **({"group": self.group} if self.group is not None else {}),
            "force_policy": "actor",
            "force_enabled": self.force_enabled,
            "force_timing": {
                "finger_encode_ms": finger_ms,
                "actor_ms": (time.perf_counter() - actor_started) * 1000,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }


def _vla_observation(obs):
    return {
        key: value
        for key, value in obs.items()
        if not key.startswith("finger_") and key != "z_grip" and not key.startswith("observation.gripper_image_")
    }


def create_policy_from_config(config, *, checkpoint=None, readout=None, device="cpu", use_triton=False):
    """Build the contact service using descriptive YAML settings, without --group."""
    from openpi.models_pytorch.readout_checkpoint import readout_manifest
    from openpi.models_pytorch.readout_checkpoint import readout_vla_checkpoint
    from openpi.models_pytorch.readout_checkpoint import resolve_readout_directory
    from openpi.serving.runner import attach_readout

    validate_joint_config(config)
    settings = config.rlt.force_stage
    readout = resolve_readout_directory(readout or settings.readout_dir or config.readout_dir)
    manifest = readout_manifest(readout)
    if manifest.get("action_space") != "joint" or manifest.get("physical_joint_dim") != 7:
        raise ValueError("Force service requires a readout with seven-joint references; EE needs an adapter")
    checkpoint = readout_vla_checkpoint(checkpoint or config.rlt.vla_checkpoint, readout)
    if checkpoint is None:
        raise ValueError("Force service requires a VLA checkpoint or bundled readout VLA weights")
    actor = None
    finger_input = None
    enabled = settings.policy == "actor" and config.rlt.force.enabled
    actor_metadata = {}
    if settings.policy == "actor":
        saved = torch.load(
            config.rlt.actor_checkpoint or settings.checkpoint_path, map_location="cpu", weights_only=True
        )
        if saved["config"]["zero_grip"] != (not enabled):
            raise ValueError("Force actor checkpoint differs from force.enabled")
        network = NetworkConfig(**saved["config"]["network"])
        token = manifest["readout_config"]
        if (
            network.action_dim != 7
            or network.horizon != manifest["action_horizon"]
            or network.z_rlt_dim != token["num_rl_tokens"] * token["embed_dim"]
            or network.z_rlt_dim != config.rlt.num_tokens * config.rlt.embed_dim
            or network.z_grip_dim != config.rlt.force.feature_dim
        ):
            raise ValueError("Force actor dimensions differ from YAML/readout")
        actor = ChunkActor(network).to(device)
        actor.load_state_dict(saved["actor"], strict=True)
        actor_metadata = {
            "rlt_actor_protocol": "pytorch-force-rlt-actor-v1",
            "z_grip_dim": network.z_grip_dim,
            "prediction_horizon": network.horizon,
            "execution_horizon": network.executed_steps,
            "action_dim": network.action_dim,
        }
        force = config.rlt.force
        if enabled and force.input == "rgb":
            if force.weights is None:
                raise ValueError("Force RGB input requires rlt.force.weights")
            encoder = SharedFingerEncoder(force.weights, roi=force.roi, device=device)
            finger_input = FingerRGBInput(encoder, max_skew_s=force.max_skew_s, max_age_s=force.max_age_s)
            if finger_input.feature_dim != network.z_grip_dim:
                raise ValueError("Finger encoder/history dimension differs from force actor")
            actor_metadata.update(
                finger_input_protocol=finger_input.protocol,
                finger_history=4,
                finger_max_skew_s=force.max_skew_s,
                finger_max_age_s=force.max_age_s,
            )
    vla_policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device=device, use_triton=use_triton)
    attach_readout(vla_policy, config, readout, require_reference_anchor=actor is not None)
    metadata = {**vla_policy.metadata, **actor_metadata, "force_policy": settings.policy, "force_enabled": enabled}
    if actor is not None:
        metadata.update(rlt_protocol=manifest["format"], require_reference_anchor=True)
    return ForceRLTPolicy(vla_policy, actor, force_enabled=enabled, finger_input=finger_input, metadata=metadata)


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
    config = train_config.get_config(args.config)
    from openpi.models_pytorch.readout_checkpoint import readout_manifest
    from openpi.models_pytorch.readout_checkpoint import readout_vla_checkpoint
    from openpi.models_pytorch.readout_checkpoint import resolve_readout_directory

    args.readout = resolve_readout_directory(args.readout)
    manifest = readout_manifest(args.readout)
    if manifest["format"] != "pytorch-rl-token-v1" or manifest["config"] != config.name:
        raise ValueError("Readout manifest protocol/config mismatch")
    if not manifest.get("camera_protocol"):
        raise ValueError("Readout camera protocol is missing")
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
    checkpoint = readout_vla_checkpoint(args.vla_checkpoint, args.readout)
    vla_policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device=args.device)
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
