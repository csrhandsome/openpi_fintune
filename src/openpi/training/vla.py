"""PyTorch VLA training implementation; invoked by scripts/train.py."""

import contextlib
import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

# The shared data transforms still import JAX; keep it off the training GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import safetensors
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.base_weights import ensure_pytorch_base_weights
from openpi.training.validation import validate_dataset


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=_config.to_dict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig, *, batch_size: int | None = None):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, shuffle=True, batch_size=batch_size)
    return data_loader, data_loader.data_config()


def move_tree_to_device_dtype(tree, device, dtype):
    """Move a batch to a device and cast floating tensors to the model dtype."""
    if isinstance(tree, torch.Tensor):
        return tree.to(device=device, dtype=dtype if tree.is_floating_point() else None)
    if dataclasses.is_dataclass(tree):
        return dataclasses.replace(
            tree,
            **{
                field.name: move_tree_to_device_dtype(getattr(tree, field.name), device, dtype)
                for field in dataclasses.fields(tree)
            },
        )
    if isinstance(tree, dict):
        return {key: move_tree_to_device_dtype(value, device, dtype) for key, value in tree.items()}
    if isinstance(tree, tuple | list):
        return type(tree)(move_tree_to_device_dtype(value, device, dtype) for value in tree)
    return tree


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save only the metadata needed to resume the Torch run.
        metadata = {
            "global_step": global_step,
            "config": _config.to_dict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        _config.save_config(config, tmp_ckpt_dir)

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")
        del model_to_save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, strict=True, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    if config.mode != "vla":
        raise ValueError("VLA backend requires mode=vla")
    if type(config.model) is not openpi.models.pi0_config.Pi0Config:
        raise ValueError(f"Torch training supports Pi0Config only; got {type(config.model).__name__}.")
    model_cfg = dataclasses.replace(config.model, dtype=config.pytorch_training_precision)
    validate_dataset(config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _config.gradient_accumulation(config, world_size)
    if (
        not config.resume
        and not config.overwrite
        and config.checkpoint_dir.exists()
        and any(config.checkpoint_dir.iterdir())
    ):
        raise FileExistsError(
            f"Checkpoint output exists: {config.checkpoint_dir}; choose a new exp_name or set resume/overwrite"
        )

    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    if not resuming and config.requires_pytorch_weights:
        weights = ensure_pytorch_base_weights(config, is_main=is_main)
        if config.pytorch_weight_path is None:
            config = dataclasses.replace(config, pytorch_weight_path=os.path.dirname(weights))
    base_model_path = None
    if config.pytorch_weight_path is not None and not resuming:
        base_model_path = (
            os.path.join(config.pytorch_weight_path, "model.safetensors")
            if os.path.isdir(config.pytorch_weight_path)
            else config.pytorch_weight_path
        )
        if not os.path.isfile(base_model_path):
            raise FileNotFoundError(f"PyTorch base weights not found: {base_model_path}")
        with safetensors.safe_open(base_model_path, framework="pt", device="cpu") as weights:
            keys = weights.keys()
            if not any(key.startswith("llm.") for key in keys) or not any(key.startswith("img.") for key in keys):
                raise ValueError(f"Expected RLinf-layout Pi0 weights with llm.* and img.* keys: {base_model_path}")

    if not resuming and config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        _config.save_config(config, config.checkpoint_dir)
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # One optimizer step covers batch_size samples. Each forward uses micro_batch_size per GPU.
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    micro_batch, accumulation = _config.gradient_accumulation(config, world_size)
    logging.info(
        f"Using batch size per GPU: {micro_batch}, accumulation={accumulation}, "
        f"global batch={config.batch_size}, world_size={world_size}"
    )

    loader, data_config = build_datasets(config, batch_size=micro_batch * world_size)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, shuffle=False, batch_size=micro_batch * world_size)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    # RLinf keeps the flow/action projections in fp32. The Gemma/SigLIP
    # backbone is converted separately after loading the base checkpoint.
    weight_dtype = torch.float32
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

    enable_gradient_checkpointing = False  # RLinf SFT runs without activation checkpointing.

    # Load RLinf-layout base weights before applying its mixed-precision policy.
    if base_model_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        state_dict = safetensors.torch.load_file(base_model_path, device="cpu")
        model.load_weights(state_dict)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    model.to_bfloat16_for_selected_params(model_cfg.dtype)
    model.to(device=device)
    model.apply_lora_freeze()

    # Log initial memory usage after model creation and cleanup.
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Keep the allocator tuning for large jobs, while TF32 is enabled above for
    # every CUDA run as in the embodied training path.
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optim = torch.optim.AdamW(
        trainable_parameters,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, micro_batch_size={micro_batch}, "
            f"accumulation={accumulation}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, trainable_params={len(trainable_parameters)}, "
            f"weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    data_iter = iter(loader)
    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)
        logged_loss = 0.0
        for micro_index in range(accumulation):
            try:
                observation, actions = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                observation, actions = next(data_iter)
            observation = move_tree_to_device_dtype(observation, device, weight_dtype)
            actions = actions.to(device=device, dtype=weight_dtype)
            losses = model(observation, actions)
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            raw_loss = losses.mean()
            logged_loss += float(raw_loss.detach())
            # All-reduce only on the last micro-batch so DDP averages the full step once.
            sync = model.no_sync() if use_ddp and micro_index + 1 < accumulation else contextlib.nullcontext()
            with sync:
                (raw_loss / accumulation).backward()

        loss = logged_loss / accumulation

        # Log memory usage after backward pass
        if global_step < 5 and is_main and torch.cuda.is_available():
            log_memory_usage(device, global_step, "after_backward")

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, max_norm=config.optimizer.clip_gradient_norm
        )

        # Optimizer step
        optim.step()
        optim.zero_grad(set_to_none=True)

        # Clear gradients more aggressively
        for param in trainable_parameters:
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

        # Collect stats
        if is_main:
            infos.append(
                {
                    "loss": loss,
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
            )

        if is_main and (global_step % config.log_interval == 0):
            elapsed = time.time() - start_time

            # Average stats over log interval
            avg_loss = sum(info["loss"] for info in infos) / len(infos)
            avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

            avg_grad_norm = None
            if any("grad_norm" in info for info in infos):
                vals = [info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
                if len(vals) > 0:
                    avg_grad_norm = sum(vals) / len(vals)
            logging.info(
                f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                if avg_grad_norm is not None
                else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
            )

            # Log to wandb
            if config.wandb_enabled and len(infos) > 0:
                log_payload = {
                    "loss": avg_loss,
                    "learning_rate": avg_lr,
                    "step": global_step,
                    "time_per_step": elapsed / config.log_interval,
                }
                if avg_grad_norm is not None:
                    log_payload["grad_norm"] = avg_grad_norm
                wandb.log(log_payload, step=global_step)

            start_time = time.time()
            infos = []  # Reset stats collection

        global_step += 1
        # Save checkpoint using the new mechanism
        save_checkpoint(model, optim, global_step, config, is_main, data_config)

        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {"loss": f"{loss:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
            )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main(config: _config.TrainConfig | None = None):
    init_logging()
    train_loop(config if config is not None else _config.cli())
