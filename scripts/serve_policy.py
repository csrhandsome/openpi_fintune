import dataclasses
import enum
import logging
import os
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8001
    # Record the policy's behavior for debugging.
    record: bool = False

    # Use an alternate audio encoder instead of Whisper for Pi0Audio configs.
    qwen: bool = False
    qwen_model_name: str = "Qwen/Qwen3-ASR-0.6B"
    qwen_device: str = "auto"
    usad: bool = False
    usad_model_name: str = "MIT-SLS/USAD2-Large"
    usad_device: str = "auto"

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.qwen and args.usad:
                raise ValueError("--qwen and --usad cannot both be enabled.")
            if args.qwen:
                train_config = _config.enable_qwen_encoder(
                    train_config,
                    qwen_model_name=args.qwen_model_name,
                    qwen_device=args.qwen_device,
                )
            elif args.qwen_device != "auto" and getattr(train_config.model, "use_qwen", False):
                # Override encoder device without --qwen, which would reset the
                # baked-in OmniAction 32s audio window.
                train_config = dataclasses.replace(
                    train_config,
                    model=dataclasses.replace(train_config.model, qwen_device=args.qwen_device),
                )
            if args.usad:
                train_config = _config.enable_usad_encoder(
                    train_config,
                    usad_model_name=args.usad_model_name,
                    usad_device=args.usad_device,
                )
            return _policy_config.create_trained_policy(
                train_config, args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _warmup_policy(policy: _policy.Policy) -> None:
    """Run one Talker + one Executor infer on the main thread before serving.

    First-call Qwen load and JAX compile inside asyncio.to_thread is what has
    been aborting this WSL box (`free(): invalid pointer` / xla_cuda_plugin).
    """
    zeros_audio = np.zeros((16_000,), dtype=np.float32)
    obs = {
        "observation/state": np.zeros((8,), dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "",
        "observation/instruction_audio": zeros_audio,
        "observation/instruction_audio_sample_rate": np.int32(16_000),
        "observation/instruction_audio_mask": np.bool_(True),
        "observation/ambient_audio": zeros_audio,
        "observation/ambient_audio_sample_rate": np.int32(16_000),
        "observation/ambient_audio_mask": np.bool_(True),
        "dialogue_prompt": "",
        "dialogue_generate": np.bool_(True),
    }
    logging.info("Warming up policy on the main thread (dialogue generate)")
    try:
        policy.infer(obs)
        logging.info("Warmup dialogue generate done")
    except Exception:
        logging.exception("Warmup dialogue generate failed; continuing")
    logging.info("Warming up policy on the main thread (action sample)")
    try:
        obs["dialogue_generate"] = np.bool_(False)
        policy.infer(obs)
        logging.info("Warmup action sample done")
    except Exception:
        logging.exception("Warmup action sample failed; continuing")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    if os.environ.get("OPENPI_WARMUP", "0") == "1":
        _warmup_policy(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
