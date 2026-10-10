"""Consistent physical/normalized action conversion for EE and joint replay."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ActionAdapter:
    action_dim: int
    representation: str = "absolute"
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    def __post_init__(self):
        if self.action_dim <= 0 or self.representation not in {"absolute", "ee_delta"}:
            raise ValueError("Invalid action representation")
        if self.representation == "ee_delta" and self.action_dim % 7:
            raise ValueError("EE delta requires one 6D pose plus gripper block per arm")
        if (self.q01 is None) != (self.q99 is None):
            raise ValueError("Normalization needs both quantiles")
        if self.q01 is not None:
            if self.q01.shape != (self.action_dim,) or self.q99.shape != (self.action_dim,):
                raise ValueError("Action quantiles have incorrect dimensions")
            if not np.isfinite(self.q01).all() or not np.isfinite(self.q99).all() or np.any(self.q99 < self.q01):
                raise ValueError("Action quantiles must be finite and ordered")

    @classmethod
    def from_stats(cls, path: str | Path | None, *, action_dim: int, representation: str = "absolute"):
        if path is None:
            return cls(action_dim, representation)
        payload = json.loads(Path(path).read_text())
        stats = payload.get("norm_stats", payload)["actions"]
        return cls(
            action_dim,
            representation,
            np.asarray(stats["q01"], np.float32)[:action_dim],
            np.asarray(stats["q99"], np.float32)[:action_dim],
        )

    def normalize(self, actions: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        result = np.asarray(actions, dtype=np.float32).copy()
        if self.representation == "ee_delta":
            blocks = result.reshape(*result.shape[:-1], -1, 7)
            state = np.asarray(proprio).reshape(*proprio.shape[:-1], 1, -1, 7)
            blocks[..., :6] -= state[..., :6]
            blocks[..., 3:6] = (blocks[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
        if self.q01 is not None:
            result = (result - self.q01) / (self.q99 - self.q01 + 1e-6) * 2 - 1
        return result

    def denormalize(self, actions: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        result = np.asarray(actions, dtype=np.float32).copy()
        if self.q01 is not None:
            result = (result + 1) * 0.5 * (self.q99 - self.q01 + 1e-6) + self.q01
        if self.representation == "ee_delta":
            blocks = result.reshape(*result.shape[:-1], -1, 7)
            state = np.asarray(proprio).reshape(*proprio.shape[:-1], 1, -1, 7)
            blocks[..., :6] += state[..., :6]
            blocks[..., 3:6] = (blocks[..., 3:6] + np.pi) % (2 * np.pi) - np.pi
        return result

    def denormalize_tensor(self, actions, proprio):
        """Differentiable physical targets for the reference smoothness loss."""
        result = actions
        if self.q01 is not None:
            q01 = torch.as_tensor(self.q01, device=actions.device, dtype=actions.dtype)
            q99 = torch.as_tensor(self.q99, device=actions.device, dtype=actions.dtype)
            result = (result + 1) * 0.5 * (q99 - q01 + 1e-6) + q01
        if self.representation == "ee_delta":
            blocks = result.reshape(*result.shape[:-1], -1, 7)
            state = proprio.reshape(*proprio.shape[:-1], 1, -1, 7)
            pose = blocks[..., :6] + state[..., :6]
            pose = torch.cat([pose[..., :3], (pose[..., 3:6] + torch.pi) % (2 * torch.pi) - torch.pi], dim=-1)
            result = torch.cat([pose, blocks[..., 6:]], dim=-1).reshape_as(result)
        return result
