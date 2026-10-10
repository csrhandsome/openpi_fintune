"""Episode RPCs carried by the existing WebSocket infer request envelope."""

import dataclasses
import logging

import numpy as np

from openpi.rlt.episode import RLTTransition
from openpi.rlt.episode import finite_array
from openpi.rlt.online import RolloutPhaseController
from openpi.rlt.settings import boolean

logger = logging.getLogger(__name__)


class OnlineRLTPolicy:
    """One active robot episode; finalized, labeled episodes enter replay atomically."""

    def __init__(self, vla_policy, service, config, *, layout=None, finger_input=None):
        self.vla_policy, self.service, self.config = vla_policy, service, config
        self.layout, self.finger_input = layout, finger_input
        self.spec = service.learner.spec
        self.phase = RolloutPhaseController(config.mode)
        self.episode_id = None
        self.policy_mode = "actor"
        self.step_id = 0
        self.pending = None
        self.cached = None
        self.drafts = []
        self.ending = False
        self._metadata = {
            **vla_policy.metadata,
            "rlt_online_protocol": "openpi-rlt-online-v1",
            "rlt_mode": config.mode,
            "force_enabled": config.force.enabled,
            "z_grip_dim": self.spec.grip_dim,
            "execution_horizon": self.spec.chunk_len,
            "critical_phase_only": config.critical_phase_only,
            "collect_enabled": config.collect.enabled,
        }
        if finger_input is not None:
            self._metadata["finger_input_protocol"] = finger_input.protocol

    @property
    def metadata(self):
        return self._metadata

    def status(self):
        return {
            **self.service.status(),
            "episode_id": self.episode_id,
            "episode_phase": self.phase.episode_phase,
            "mode": self.config.mode,
            "collect_enabled": self.config.collect.enabled,
        }

    def infer(self, request):
        event = request.get("rlt_event", "infer")
        if event == "status":
            return self.status()
        if event == "begin_episode":
            return self._begin(request)
        self._check_episode(request)
        if event == "infer":
            return self._infer(request)
        if event == "observe":
            return self._observe(request)
        if event == "end_episode":
            return self._end(request)
        if event == "abort_episode":
            self.reset()
            return self.status()
        raise ValueError(f"Unknown rlt_event: {event}")

    def _check_episode(self, request):
        if self.episode_id is None or request.get("episode_id") != self.episode_id:
            raise ValueError("An accepted begin_episode with matching episode_id is required")

    def _begin(self, request):
        episode_id = request.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("A unique nonempty episode_id is required")
        policy_mode = request.get("policy_mode", "actor")
        if policy_mode not in {"base", "actor"}:
            raise ValueError("policy_mode must be base or actor")
        status = self.service.status()
        if status["learner_error"]:
            raise RuntimeError(f"Learner failed: {status['learner_error']}")
        if self.service.journal is not None and self.service.journal.episode_metadata(episode_id) is not None:
            raise ValueError("Episode ID already committed; use a unique episode_id")
        phase = self.phase.begin_episode(status)
        if phase == "warmup_wait_online":
            return {**self.status(), "accepted": False, "phase": phase}
        self.episode_id, self.policy_mode = episode_id, policy_mode
        logger.info("Begin episode=%s phase=%s policy_mode=%s", episode_id, phase, policy_mode)
        return {**self.status(), "accepted": True, "phase": phase}

    def _features(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("observation must be a mapping")
        if "observation.proprio" in observation:
            proprio = observation["observation.proprio"]
        elif self.layout is not None:
            proprio = self.layout.combine(
                observation[f"observation.{self.layout.target}"], observation["observation.gripper_position"]
            )
        else:
            raise ValueError("Provide observation.proprio in the physical robot state layout")
        feature = {"proprio": finite_array(proprio, (self.spec.proprio_dim,), "proprio")}
        if self.spec.grip_dim:
            grip = self.finger_input(observation) if self.finger_input else observation.get("z_grip")
            feature["z_grip"] = finite_array(grip, (self.spec.grip_dim,), "z_grip")
        vla_obs = {
            key: value
            for key, value in observation.items()
            if not key.startswith("finger_")
            and not key.startswith("observation.gripper_image_")
            and key not in {"z_grip", "observation.proprio"}
        }
        result = self.vla_policy.infer(vla_obs)
        feature["z_rlt"] = finite_array(result["z_rlt"], (self.spec.z_dim,), "z_rlt")
        feature["ref_chunk"] = finite_array(result["ref_chunk"], (self.spec.horizon, self.spec.action_dim), "ref_chunk")
        return result, feature

    def _infer(self, request):
        if self.pending is not None and self.config.collect.enabled:
            raise ValueError("Send execution feedback before requesting another action")
        if self.ending:
            raise ValueError("A partial execution must end the episode")
        if self.config.critical_phase_only:
            critical = request.get("in_critical_phase")
            boolean("in_critical_phase", critical)
        else:
            critical = True
        if self.cached is not None:
            if "observation" in request:
                raise ValueError("After observe, infer without observation to use the aligned next-state reference")
            result, feature = self.cached
        else:
            result, feature = self._features(request.get("observation"))
        source, version = "BASE", -1
        actions = feature["ref_chunk"].copy()
        if self.phase.episode_phase in {"online", "eval"} and critical and self.policy_mode == "actor":
            if self.service.status()["learner_error"]:
                raise RuntimeError("Learner failed; abort this episode before resuming")
            actions, version = self.service.propose(feature)
            actions = finite_array(actions, (self.spec.horizon, self.spec.action_dim), "actor actions")
            source = "RL"
        step_id = self.step_id
        self.step_id += 1
        self.pending = {"feature": feature, "source": source, "version": version, "step_id": step_id}
        self.cached = None
        return {
            **result,
            "actions": actions,
            "refined_chunk": actions.copy(),
            "source": source,
            "actor_version": version,
            "episode_id": self.episode_id,
            "step_id": step_id,
            "episode_phase": self.phase.episode_phase,
            "execution_horizon": self.spec.chunk_len,
        }

    def _observe(self, request):
        if not self.config.collect.enabled:
            raise ValueError("Execution feedback is only required when collection is enabled")
        if (
            self.pending is None
            or type(request.get("step_id")) is not int
            or request.get("step_id") != self.pending["step_id"]
        ):
            raise ValueError("Execution feedback must match the pending step_id")
        actual = np.asarray(request.get("executed_actions"), dtype=np.float32)
        if actual.ndim != 2 or not 0 < len(actual) <= self.spec.chunk_len:
            raise ValueError("executed_actions must contain 1..execution_horizon acknowledged targets")
        actual = finite_array(actual, (len(actual), self.spec.action_dim), "executed_actions")
        source = request.get("source", self.pending["source"])
        if source not in {self.pending["source"], "HUMAN"}:
            raise ValueError("Feedback source must match the proposal or identify HUMAN intervention")
        next_result, next_feature = self._features(request.get("next_observation"))
        count = len(actual)
        executed = np.repeat(actual[-1][None], self.spec.chunk_len, axis=0)
        executed[:count] = actual
        mask = np.arange(self.spec.chunk_len) < count
        sources = np.full(self.spec.chunk_len, "PAD", dtype="U5")
        sources[:count] = source
        current = self.pending["feature"]
        record = RLTTransition(
            episode_id=self.episode_id,
            step_id=self.pending["step_id"],
            z_rlt=current["z_rlt"],
            proprio=current["proprio"],
            ref_chunk=current["ref_chunk"],
            action_chunk=executed,
            rewards=np.zeros(self.spec.chunk_len, np.float32),
            executed_mask=mask,
            done=False,  # Finalized once the episode has a success/failure label.
            next_z_rlt=next_feature["z_rlt"],
            next_proprio=next_feature["proprio"],
            next_ref_chunk=next_feature["ref_chunk"],
            source_chunk=sources,
            success=False,
            z_grip=current.get("z_grip"),
            next_z_grip=next_feature.get("z_grip"),
        )
        self.drafts.append((record, self.pending["version"]))
        self.pending = None
        self.cached = (next_result, next_feature)
        # End at a partial execution or intervention boundary; never cross it.
        self.ending = count < self.spec.chunk_len or source == "HUMAN"
        return {**self.status(), "observed_step_id": record.step_id}

    def _end(self, request):
        success = request.get("success")
        boolean("success", success)
        if self.config.collect.enabled:
            if self.pending is not None or not self.drafts:
                raise ValueError("An episode requires acknowledged execution and a final next_observation")
            total = sum(int(record.executed_mask.sum()) for record, _ in self.drafts)
            rewards = self.config.reward.generate(total, success=success)
            records, offset = [], 0
            for index, (record, _) in enumerate(self.drafts):
                count = int(record.executed_mask.sum())
                chunk_rewards = np.zeros(self.spec.chunk_len, np.float32)
                chunk_rewards[:count] = rewards[offset : offset + count]
                offset += count
                records.append(
                    dataclasses.replace(
                        record, rewards=chunk_rewards, success=success, done=index == len(self.drafts) - 1
                    )
                )
            self.service.commit_episode(
                records,
                {
                    "success": success,
                    "phase": self.phase.episode_phase,
                    "actor_versions": [version for _, version in self.drafts],
                    "force_enabled": self.config.force.enabled,
                },
            )
        completed_id = self.episode_id
        self.reset()
        return {**self.status(), "completed_episode_id": completed_id, "success": success}

    def reset(self):
        self.phase.finish_episode()
        self.episode_id = None
        self.step_id = 0
        self.pending = None
        self.cached = None
        self.drafts = []
        self.ending = False

    def close(self):
        self.service.close()
