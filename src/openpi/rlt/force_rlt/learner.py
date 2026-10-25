"""Short-chunk PyTorch actor-critic: H-step proposals, E-step TD targets."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from openpi.rlt.force_rlt.networks import ChunkActor
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.rlt.force_rlt.networks import TwinCritic
from openpi.rlt.force_rlt.replay import Transition


@dataclass(frozen=True)
class LearnerConfig:
    network: NetworkConfig
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    beta_bc: float = 1.0
    beta_smooth: float = 0.0
    zero_grip: bool = False

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Gamma and target update rate must be in (0,1]")
        if min(self.actor_lr, self.critic_lr) <= 0 or min(self.beta_bc, self.beta_smooth) < 0:
            raise ValueError("Invalid optimizer or regularization settings")


def discounted_rewards(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    discounts = gamma ** torch.arange(rewards.shape[1], device=rewards.device)
    return (rewards * mask * discounts[None]).sum(dim=1)


def td_target(
    rewards: torch.Tensor, mask: torch.Tensor, done: torch.Tensor, next_q: torch.Tensor, gamma: float
) -> torch.Tensor:
    count = mask.sum(dim=1)
    return discounted_rewards(rewards, mask, gamma) + (~done).float() * gamma**count * next_q


class ChunkLearner:
    def __init__(self, config: LearnerConfig, *, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.actor = ChunkActor(config.network).to(device)
        self.critic = TwinCritic(config.network).to(device)
        self.target_actor = ChunkActor(config.network).to(device)
        self.target_critic = TwinCritic(config.network).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=config.critic_lr)
        self.steps = 0

    def _tensor(self, values, *, boolean=False):
        return torch.as_tensor(np.stack(values), dtype=torch.bool if boolean else torch.float32, device=self.device)

    def _state(self, records: list[Transition], *, next_state=False):
        states = [record.next_state if next_state else record.state for record in records]
        grip = self._tensor([state.z_grip for state in states])
        if self.config.zero_grip:
            grip = torch.zeros_like(grip)
        return (
            self._tensor([state.z_rlt for state in states]),
            grip,
            self._tensor([state.q for state in states]),
            self._tensor([state.ref_q for state in states]),
        )

    def update(self, records: list[Transition]) -> dict[str, float]:
        if not records:
            raise ValueError("Empty RL batch")
        cfg = self.config.network
        for record in records:
            record.validate(cfg)
        state = self._state(records)
        next_state = self._state(records, next_state=True)
        actions = self._tensor([record.executed_q for record in records])
        mask = self._tensor([record.executed_mask for record in records], boolean=True)
        human = self._tensor([record.action_sources == "HUMAN" for record in records], boolean=True) & mask
        rewards = self._tensor([record.rewards for record in records])
        done = torch.as_tensor([record.done for record in records], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_action = self.target_actor.sample(*next_state)
            next_mask = torch.ones_like(mask)
            tq1, tq2 = self.target_critic(*next_state, next_action[:, : cfg.executed_steps], next_mask)
            target = td_target(rewards, mask, done, torch.minimum(tq1, tq2), self.config.gamma)
        q1, q2 = self.critic(*state, actions, mask)
        critic_loss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad = False
        proposal = self.actor.sample(*state)
        proposal_q, _ = self.critic(*state, proposal[:, : cfg.executed_steps], mask)
        # Supervise acknowledged human targets, retaining VLA references elsewhere.
        bc_target = state[3].clone()
        bc_target[:, : cfg.executed_steps] = torch.where(human[..., None], actions, bc_target[:, : cfg.executed_steps])
        bc_mask = torch.ones(proposal.shape[:2], dtype=torch.bool, device=self.device)
        bc_mask[:, : cfg.executed_steps] = mask
        bc = ((proposal - bc_target).square().mean(dim=-1) * bc_mask).sum() / bc_mask.sum()
        pair_mask = bc_mask[:, 1:] & bc_mask[:, :-1]
        smooth = (
            (
                (proposal[:, 1:] - proposal[:, :-1] - (bc_target[:, 1:] - bc_target[:, :-1])).square().mean(dim=-1)
                * pair_mask
            ).sum()
            / pair_mask.sum().clamp_min(1)
            if cfg.horizon > 1
            else proposal.new_zeros(())
        )
        actor_loss = -proposal_q.mean() + self.config.beta_bc * bc + self.config.beta_smooth * smooth
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad = True

        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                self.target_actor.parameters(), self.actor.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, self.config.tau)
            for target_parameter, source_parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(source_parameter, self.config.tau)
        self.steps += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "bc": float(bc.detach()),
            "smooth": float(smooth.detach()),
            "human_fraction": float(human.sum() / mask.sum()),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        saved = torch.load(path, map_location=device, weights_only=True)
        network = NetworkConfig(**saved["config"].pop("network"))
        config = LearnerConfig(network=network, **saved["config"])
        learner = cls(config, device=device)
        for name in ("actor", "critic", "target_actor", "target_critic"):
            getattr(learner, name).load_state_dict(saved[name])
        for name in ("actor_opt", "critic_opt"):
            getattr(learner, name).load_state_dict(saved[name])
        learner.steps = saved["steps"]
        return learner
