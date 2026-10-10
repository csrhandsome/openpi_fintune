"""Build VLA or online/eval RLT servers from inference YAMLs."""

import dataclasses
import logging
import pathlib

from safetensors.torch import load_model
import torch
import yaml

from openpi.models_pytorch.readout_checkpoint import readout_manifest
from openpi.models_pytorch.readout_checkpoint import readout_vla_checkpoint
from openpi.models_pytorch.readout_checkpoint import resolve_readout_directory
from openpi.models_pytorch.rl_token import RLTokenConfig
from openpi.models_pytorch.rl_token import RLTokenReadout
from openpi.policies import policy as policy_module
from openpi.policies import policy_config
from openpi.serving import config as serving_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as training_config


@dataclasses.dataclass
class Args:
    config: str = "config/server/openpi.yaml"
    checkpoint: str | None = None
    readout: str | None = None
    device: str | None = None
    host: str | None = None
    port: int | None = None
    record: bool | None = None


def load_settings(args):
    path = serving_config.find_config(args.config)
    if not path.is_file():
        path = pathlib.Path(training_config.get_config(args.config).config_path)
    # Keep explicit training YAMLs working for offline replay and existing clients.
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(values, dict) and "train_config" in values:
        settings = serving_config.load_config(str(path))
    else:
        training = training_config.get_config(str(path))
        force_stage = training.mode == "rlt" and training.rlt.stage == "force"
        settings = serving_config.InferenceConfig(train_config=str(path), mode="force" if force_stage else "vla")
    overrides = {
        key: getattr(args, key)
        for key in ("checkpoint", "readout", "device", "host", "port", "record")
        if getattr(args, key) is not None
    }
    return dataclasses.replace(settings, **overrides)


def latest_checkpoint(config: training_config.TrainConfig) -> pathlib.Path:
    root = config.checkpoint_dir
    candidates = (
        [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.isdecimal() and (path / "model.safetensors").is_file()
        ]
        if root.is_dir()
        else []
    )
    if not candidates:
        raise FileNotFoundError(f"No PyTorch checkpoint in {root}; set --checkpoint or train this YAML first")
    return max(candidates, key=lambda path: int(path.name))


def attach_readout(policy, config, directory, *, require_reference_anchor=False):
    manifest = readout_manifest(directory)
    if manifest["format"] != "pytorch-rl-token-v1" or manifest["config"] != config.name:
        raise ValueError("Readout manifest protocol/config mismatch")
    if (config.model.action_horizon, config.model.action_dim) != (
        manifest["action_horizon"],
        manifest["model_action_dim"],
    ):
        raise ValueError("Readout VLA dimensions disagree with loaded YAML")
    metadata = config.policy_metadata or {}
    for key in ("action_space", "action_representation"):
        if key in manifest and key in metadata and manifest[key] != metadata[key]:
            raise ValueError(f"Readout {key} differs from YAML")
    if manifest.get("physical_action_dim", metadata.get("action_dim")) != metadata.get("action_dim"):
        raise ValueError("Readout physical action dimension differs from YAML")
    readout = RLTokenReadout(RLTokenConfig(**manifest["readout_config"]))
    load_model(readout, directory / "readout.safetensors", strict=True)
    policy.attach_rl_readout(
        readout,
        metadata={
            # data_collect routes rlt_protocol to its contact-actor client.
            # A standalone readout still executes ordinary VLA actions.
            "rlt_readout_protocol": manifest["format"],
            "camera_protocol": manifest["camera_protocol"],
            "prefix_shape": manifest["prefix_shape"],
            "z_rlt_dim": readout.config.num_rl_tokens * readout.config.embed_dim,
            "require_reference_anchor": require_reference_anchor,
        },
    )


def create_policy(args: Args):
    settings = load_settings(args)
    config = training_config.get_config(settings.train_config)
    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if settings.mode == "force":
        from openpi.rlt.force_rlt.serve import create_policy_from_config

        return create_policy_from_config(
            config, checkpoint=settings.checkpoint, readout=settings.readout, device=device
        )
    checkpoint = settings.checkpoint
    if checkpoint is None and config.mode == "rlt":
        checkpoint = config.rlt.vla_checkpoint
    checkpoint = training_config.local_path(checkpoint) if checkpoint else None
    if config.mode == "rlt" or settings.readout is not None or settings.mode == "rlt":
        directory = resolve_readout_directory(
            training_config.local_path(settings.readout) if settings.readout else config.readout_dir
        )
        checkpoint = readout_vla_checkpoint(checkpoint, directory)
    if checkpoint is None:
        if config.mode == "rlt":
            raise ValueError("Set rlt.vla_checkpoint, pass --checkpoint, or use a readout with bundled VLA weights")
        checkpoint = latest_checkpoint(config)
    policy = policy_config.create_trained_policy(config, checkpoint, pytorch_device=device)
    if config.mode == "rlt" or settings.readout is not None or settings.mode == "rlt":
        attach_readout(policy, config, directory)
    if settings.mode == "rlt":
        policy = create_rlt_policy(policy, config, settings.rlt, device, checkpoint, directory)
    return policy


def create_rlt_policy(policy, training, settings, device, checkpoint, readout):
    from openpi.rlt.action_representation import ActionAdapter
    from openpi.rlt.episode import ReplaySpec
    from openpi.rlt.features import frozen_model_identity
    from openpi.rlt.force_rlt.finger_input import FingerRGBInput
    from openpi.rlt.force_rlt.tactile_encoder import SharedFingerEncoder
    from openpi.rlt.learner import ChunkLearner
    from openpi.rlt.online import OnlineLearner
    from openpi.rlt.replay import ReplayJournal
    from openpi.serving.rlt_policy import OnlineRLTPolicy

    metadata = policy.metadata
    spec = ReplaySpec(
        z_dim=metadata["z_rlt_dim"],
        proprio_dim=settings.proprio_dim or metadata["action_dim"],
        horizon=metadata["action_horizon"],
        chunk_len=settings.chunk_len,
        action_dim=metadata["action_dim"],
        action_space=metadata["action_space"],
        fps=metadata["action_fps"],
        grip_dim=settings.force.grip_dim,
    )
    finger_input = None
    if settings.force.enabled and settings.force.input == "rgb":
        if settings.force.weights is None:
            raise ValueError("force.enabled with RGB input requires force.weights")
        encoder = SharedFingerEncoder(settings.force.weights, roi=settings.force.roi, device=device)
        finger_input = FingerRGBInput(encoder, max_skew_s=settings.force.max_skew_s, max_age_s=settings.force.max_age_s)
    model_id = frozen_model_identity(training.config_path, metadata, checkpoint, readout, force=settings.force)
    learner = (
        ChunkLearner.load(settings.actor_checkpoint, device=device)
        if settings.actor_checkpoint
        else ChunkLearner(
            spec,
            settings.learner.training_config(),
            actor_config=settings.actor,
            critic_config=settings.critic,
            adapter=ActionAdapter.from_stats(
                settings.action_norm_stats, action_dim=spec.action_dim, representation=settings.action_representation
            ),
            device=device,
        )
    )
    if learner.spec != spec:
        raise ValueError("Actor checkpoint differs from inference force/state/action layout")
    if settings.mode == "online_rl" and learner.config.actor_update_period != settings.learner.actor_update_period:
        raise ValueError("Online resume requires the checkpoint actor_update_period")
    if settings.mode == "online_rl" and learner.config.optimizer != settings.learner.optimizer:
        raise ValueError("Online resume requires the checkpoint optimizer kind")
    prior_id = learner.runtime_state.get("model_id")
    if prior_id is not None and prior_id != model_id:
        raise ValueError("Actor checkpoint frozen VLA/readout/force identity differs from inference")
    learner.runtime_state["model_id"] = model_id
    if settings.mode == "online_rl" and not settings.actor_checkpoint:
        output = pathlib.Path(settings.learner.checkpoint_path)
        if output.exists():
            raise FileExistsError(f"Actor checkpoint exists: {output}; set rlt.actor_checkpoint to resume")
    journal = None
    if settings.collect.enabled or settings.mode == "online_rl":
        if not settings.collect.enabled and not (pathlib.Path(settings.replay.directory) / "manifest.json").exists():
            raise ValueError("Collection disabled: an existing replay directory is required")
        journal = ReplayJournal(settings.replay.directory, spec)
        journal.bind_provenance({"model_id": model_id, "action_source": "command"})
    for optimizer, lr in (
        (learner.actor_opt, settings.learner.actor_lr),
        (learner.critic_opt, settings.learner.critic_lr),
    ):
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = settings.learner.weight_decay
    layout = getattr(training.data, "layout", None)
    service = OnlineLearner(learner, journal, settings, seed=training.seed)
    return OnlineRLTPolicy(policy, service, settings, layout=layout, finger_input=finger_input)


def serve(args: Args):
    settings = load_settings(args)
    policy = create_policy(args)
    runtime = policy
    metadata = policy.metadata
    if settings.record:
        policy = policy_module.PolicyRecorder(policy, "policy_records")
    logging.basicConfig(level=logging.INFO, force=True)
    try:
        WebsocketPolicyServer(policy, host=settings.host, port=settings.port, metadata=metadata).serve_forever()
    finally:
        if hasattr(runtime, "close"):
            runtime.close()
