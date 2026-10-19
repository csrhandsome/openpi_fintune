"""Validation metrics on actions after the flow-matching sampler has finished.

Training loss is the velocity-field MSE at a random time. These metrics instead
integrate the field from standard Gaussian noise to t=0, then compare that action
chunk with the ground truth in the normalized space the model sees. Padded action
dimensions are left out. Component values are means of the squared error.
"""

import torch
import torch.distributed as dist
import torch.nn.parallel


def component_slices(layout) -> dict[str, tuple[int, int]]:
    """Name the physical action blocks stored contiguously per arm."""
    if layout.num_arms == 1:
        prefixes = [""]
    elif layout.num_arms == 2:
        prefixes = ["left_", "right_"]
    else:
        prefixes = [f"arm{index}_" for index in range(layout.num_arms)]
    slices = {}
    for arm, prefix in enumerate(prefixes):
        start = arm * layout.arm_dim
        if layout.action_space == "joint":
            slices[f"{prefix}joint"] = (start, start + layout.motion_dim)
        else:
            slices[f"{prefix}pos"] = (start, start + 3)
            slices[f"{prefix}rot"] = (start + 3, start + 6)
        grip = start + layout.motion_dim
        slices[f"{prefix}gripper"] = (grip, grip + layout.gripper_dim)
    return slices


def squared_error_totals(prediction, target, physical_dim: int, slices: dict[str, tuple[int, int]]):
    """Sum of squared errors and element counts, ignoring padding past ``physical_dim``."""
    error = (prediction[..., :physical_dim] - target[..., :physical_dim]).to(dtype=torch.float32)
    square = torch.square(error)
    totals = {"loss": (float(square.sum().item()), int(square.numel()))}
    width = square.shape[-1]
    for name, (start, end) in slices.items():
        if start < 0 or end > width or start >= end:
            raise ValueError(f"Action slice {name}={start}:{end} is outside the {width} physical dimensions")
        part = square[..., start:end]
        totals[name] = (float(part.sum().item()), int(part.numel()))
    return totals


def average_squared_errors(totals: dict[str, tuple[float, int]], device) -> dict[str, float]:
    """Global mean of the squared error. All ranks contribute their sums and counts."""
    keys = list(totals)
    packed = torch.tensor(
        [totals[key][0] for key in keys] + [float(totals[key][1]) for key in keys],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    half = len(keys)
    metrics = {}
    for index, key in enumerate(keys):
        count = float(packed[half + index].item())
        name = "eval_loss" if key == "loss" else f"eval_l2_{key}"
        metrics[name] = float(packed[index].item() / count) if count else float("nan")
    return metrics


def denoised_action_metrics(
    model, batches, *, physical_dim: int, slices: dict[str, tuple[int, int]], num_steps: int, device
):
    """Euler-sample actions and score them against the normalized ground truth.

    ``sample_actions`` turns image augmentation off. The returned ``eval_loss`` is the
    mean squared error over the physical action dimensions. Component keys are means of
    the same squared error on each slice.
    """
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    totals = {"loss": (0.0, 0)}
    for name in slices:
        totals[name] = (0.0, 0)
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for observation, actions in batches:
                prediction = module.sample_actions(observation, num_steps=num_steps)
                for key, (total, count) in squared_error_totals(prediction, actions, physical_dim, slices).items():
                    previous_total, previous_count = totals[key]
                    totals[key] = (previous_total + total, previous_count + count)
    finally:
        model.train(training)
    return average_squared_errors(totals, device)
