"""Sparse terminal rewards from an existing human episode annotation."""

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


class EpisodeRewards(Protocol):
    def generate(self, steps: int, *, success: bool) -> np.ndarray: ...


@dataclass(frozen=True)
class TerminalReward:
    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_reward: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.success_reward, self.failure_reward, self.step_reward)):
            raise ValueError("Reward values must be finite")

    def generate(self, steps: int, *, success: bool) -> np.ndarray:
        if type(success) is not bool:
            raise ValueError("An explicit success/failure label is required; unannotated is not failure")
        if type(steps) is not int or steps <= 0:
            raise ValueError("Reward requires at least one actual transition")
        rewards = np.full(steps, self.step_reward, dtype=np.float32)
        rewards[-1] = self.success_reward if success else self.failure_reward
        return rewards
