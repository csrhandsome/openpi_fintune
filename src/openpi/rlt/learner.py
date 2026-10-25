"""PyTorch training orchestration for separately replaceable actor and critic."""

import copy
from dataclasses import asdict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional

from openpi.rlt.action_representation import ActionAdapter
from openpi.rlt.actor import ActorConfig
from openpi.rlt.actor import ChunkActor
from openpi.rlt.critic import CriticConfig
from openpi.rlt.critic import TwinCritic
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.episode import RLTTransition


@dataclass(frozen=True)
class LearnerConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    bc_weight: float = 1.0
    q_weight: float = 1.0
    reference_dropout: float = 0.5
    actor_update_period: int = 2
    actor_q_reduction: str = "q1"
    optimizer: str = "adam"
    weight_decay: float = 0.0
    target_update_period: int | None = None  # None follows actor_update_period.
    delta_weight: float = 0.0
    smooth_action_dims: int = 6

    def __post_init__(self):
        if self.actor_q_reduction not in {"q1", "min"} or self.optimizer not in {"adam", "adamw"}:
            raise ValueError("Invalid actor Q reduction or optimizer")
        if not all(math.isfinite(x) and x >= 0 for x in (self.weight_decay, self.delta_weight)):
            raise ValueError("Invalid weight decay or smoothness weight")
        if type(self.smooth_action_dims) is not int or self.smooth_action_dims <= 0:
            raise ValueError("smooth_action_dims must be positive")
        if self.target_update_period is not None and (
            type(self.target_update_period) is not int or self.target_update_period <= 0
        ):
            raise ValueError("target_update_period must be positive or None")
        if not all(
            math.isfinite(x)
            for x in (
                self.gamma,
                self.tau,
                self.actor_lr,
                self.critic_lr,
                self.bc_weight,
                self.q_weight,
                self.reference_dropout,
            )
        ):
            raise ValueError("Learner settings must be finite")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1 or not 0 <= self.reference_dropout <= 1:
            raise ValueError("Invalid discount, target update rate or reference dropout")
        if min(self.actor_lr, self.critic_lr) <= 0 or min(self.bc_weight, self.q_weight) < 0:
            raise ValueError("Invalid learning rate or actor loss weight")
        if type(self.actor_update_period) is not int or self.actor_update_period <= 0:
            raise ValueError("actor_update_period must be positive")


def td_target(rewards, mask, done, next_q, gamma):
    discounts = gamma ** torch.arange(rewards.shape[1], device=rewards.device)
    bootstrap = torch.where(done, torch.zeros_like(next_q), next_q)
    return (rewards * mask * discounts[None]).sum(dim=1) + gamma ** mask.sum(dim=1) * bootstrap


class ChunkLearner:
    def __init__(
        self,
        spec: ReplaySpec,
        config: LearnerConfig | None = None,
        *,
        actor_config: ActorConfig | None = None,
        critic_config: CriticConfig | None = None,
        adapter: ActionAdapter | None = None,
        device: str = "cpu",
    ):
        config = config or LearnerConfig()
        self.spec, self.config = spec, config
        self.device = torch.device(device)
        self.adapter = adapter or ActionAdapter(spec.action_dim)
        if self.adapter.action_dim != spec.action_dim:
            raise ValueError("Action adapter and replay dimensions differ")
        if self.adapter.representation == "ee_delta" and (
            spec.action_space != "ee" or spec.proprio_dim != spec.action_dim
        ):
            raise ValueError("EE delta requires matching EE state and action blocks")
        self.actor = ChunkActor(spec, actor_config).to(device)
        self.critic = TwinCritic(spec, critic_config).to(device)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(requires_grad=False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(requires_grad=False)
        optimizer = torch.optim.Adam if config.optimizer == "adam" else torch.optim.AdamW
        self.actor_opt = optimizer(self.actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay)
        self.critic_opt = optimizer(self.critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay)
        self.steps = 0
        self.runtime_state = {}

    def _tensor(self, values, *, dtype=torch.float32):
        return torch.as_tensor(np.stack(values), dtype=dtype, device=self.device)

    def _state(self, records, *, next_state=False):
        prefix = "next_" if next_state else ""
        proprio = np.stack([getattr(r, prefix + "proprio") for r in records])
        ref = np.stack([getattr(r, prefix + "ref_chunk") for r in records])
        return (
            self._tensor([getattr(r, prefix + "z_rlt") for r in records]),
            torch.as_tensor(proprio, device=self.device),
            torch.as_tensor(self.adapter.normalize(ref, proprio), device=self.device),
            self._tensor([getattr(r, prefix + "z_grip") for r in records]) if self.spec.grip_dim else None,
        )

    def update(self, records: list[RLTTransition]) -> dict[str, float]:
        if not records:
            raise ValueError("Empty RL batch")
        for record in records:
            record.validate(self.spec)
        current, next_state = self._state(records), self._state(records, next_state=True)
        physical = np.stack([r.action_chunk for r in records])
        actions = torch.as_tensor(
            self.adapter.normalize(physical, np.stack([r.proprio for r in records])), device=self.device
        )
        mask = self._tensor([r.executed_mask for r in records], dtype=torch.bool)
        rewards = self._tensor([r.rewards for r in records])
        done = torch.tensor([r.done for r in records], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            next_action = self.target_actor.sample(*next_state)[:, : self.spec.chunk_len]
            q1, q2 = self.target_critic(*next_state[:2], next_action, torch.ones_like(mask), next_state[3])
            target = td_target(rewards, mask, done, torch.minimum(q1, q2), self.config.gamma)
        q1, q2 = self.critic(*current[:2], actions, mask, current[3])
        critic_loss = functional.mse_loss(q1, target) + functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.steps += 1
        human = self._tensor([r.source_chunk == "HUMAN" for r in records], dtype=torch.bool) & mask
        metrics = {
            "critic_loss": float(critic_loss.detach()),
            "human_fraction": float(human.sum() / mask.sum()),
        }
        if self.steps % self.config.actor_update_period == 0:
            self.critic.requires_grad_(requires_grad=False)
            try:
                keep = torch.rand(len(records), 1, 1, device=self.device) >= self.config.reference_dropout
                predicted = self.actor.sample(*current, reference_condition=current[2] * keep)[:, : self.spec.chunk_len]
                actor_q1, actor_q2 = self.critic(*current[:2], predicted, mask, current[3])
                bc_target = torch.where(human[..., None], actions, current[2][:, : self.spec.chunk_len])
                bc = (((predicted - bc_target).square().mean(dim=-1)) * mask).sum() / mask.sum()
                actor_q = actor_q1 if self.config.actor_q_reduction == "q1" else torch.minimum(actor_q1, actor_q2)
                absolute = self.adapter.denormalize_tensor(predicted, current[1])
                absolute_target = self.adapter.denormalize_tensor(bc_target, current[1])
                dims = min(self.config.smooth_action_dims, self.spec.action_dim)
                delta_error = (
                    (
                        (absolute[:, 1:, :dims] - absolute[:, :-1, :dims])
                        - (absolute_target[:, 1:, :dims] - absolute_target[:, :-1, :dims])
                    )
                    .square()
                    .mean(dim=-1)
                )
                pair_mask = mask[:, 1:] & mask[:, :-1]
                smooth = (delta_error * pair_mask).sum() / pair_mask.sum().clamp_min(1)
                actor_loss = (
                    self.config.bc_weight * bc
                    - self.config.q_weight * actor_q.mean()
                    + self.config.delta_weight * smooth
                )
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()
                metrics.update(
                    actor_loss=float(actor_loss.detach()),
                    bc=float(bc.detach()),
                    actor_q=float(actor_q.mean().detach()),
                    delta_penalty=float(smooth.detach()),
                )
            finally:
                self.critic.requires_grad_(requires_grad=True)
        period = self.config.target_update_period or self.config.actor_update_period
        if self.steps % period == 0:
            with torch.no_grad():
                for target_net, source_net in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
                    for target_parameter, parameter in zip(
                        target_net.parameters(), source_net.parameters(), strict=True
                    ):
                        target_parameter.lerp_(parameter, self.config.tau)
        return metrics

    def propose(self, feature: dict, *, deterministic: bool = True) -> np.ndarray:
        proprio = np.asarray(feature["proprio"], dtype=np.float32)[None]
        ref = self.adapter.normalize(np.asarray(feature["ref_chunk"])[None], proprio)
        with torch.no_grad():
            proposed = (
                self.actor.sample(
                    self._tensor([feature["z_rlt"]]),
                    torch.as_tensor(proprio, device=self.device),
                    torch.as_tensor(ref, device=self.device),
                    self._tensor([feature["z_grip"]]) if self.spec.grip_dim else None,
                    deterministic=deterministic,
                )
                .cpu()
                .numpy()
            )
        return self.adapter.denormalize(proposed, proprio)[0]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "openpi-rlt-learner-v2",
            "spec": asdict(self.spec),
            "config": asdict(self.config),
            "actor_config": asdict(self.actor.config),
            "critic_config": asdict(self.critic.config),
            "adapter": {
                "action_dim": self.adapter.action_dim,
                "representation": self.adapter.representation,
                "q01": None if self.adapter.q01 is None else torch.from_numpy(self.adapter.q01),
                "q99": None if self.adapter.q99 is None else torch.from_numpy(self.adapter.q99),
            },
            "states": {
                name: getattr(self, name).state_dict()
                for name in ("actor", "critic", "target_actor", "target_critic", "actor_opt", "critic_opt")
            },
            "steps": self.steps,
            "rng_state": torch.get_rng_state(),
            "runtime_state": self.runtime_state,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["format"] != "openpi-rlt-learner-v2":
            raise ValueError("Unsupported learner checkpoint; retrain the old actor/critic architecture")
        adapter = dict(saved["adapter"])
        for key in ("q01", "q99"):
            if adapter[key] is not None:
                adapter[key] = adapter[key].numpy()
        learner = cls(
            ReplaySpec(**saved["spec"]),
            LearnerConfig(**saved["config"]),
            actor_config=ActorConfig(**saved["actor_config"]),
            critic_config=CriticConfig(**saved["critic_config"]),
            adapter=ActionAdapter(**adapter),
            device=device,
        )
        for name, state in saved["states"].items():
            getattr(learner, name).load_state_dict(state)
        learner.steps = saved["steps"]
        learner.runtime_state = saved.get("runtime_state", {})
        torch.set_rng_state(saved["rng_state"])
        return learner
