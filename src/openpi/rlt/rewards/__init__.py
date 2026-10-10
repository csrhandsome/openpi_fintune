"""Reward strategies consume episode labels, independently of replay storage."""

from openpi.rlt.rewards.terminal import EpisodeRewards
from openpi.rlt.rewards.terminal import TerminalReward

__all__ = ["EpisodeRewards", "TerminalReward"]
