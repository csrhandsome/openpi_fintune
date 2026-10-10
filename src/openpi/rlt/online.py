"""Background PyTorch learner and episode-boundary actor handoff.

The inference actor is a separate, frozen copy. Readiness includes both learner
progress and publication of a snapshot containing the warmup updates.
"""

import copy
import logging
import threading

import numpy as np
import torch

from openpi.rlt.replay import ReplayBuffer

logger = logging.getLogger(__name__)


class RolloutPhaseController:
    def __init__(self, mode):
        self.mode = mode
        self.episode_phase = None

    def begin_episode(self, status):
        if self.episode_phase is not None:
            raise ValueError("Finish or abort the current episode first")
        if self.mode == "eval":
            phase = "eval"
        elif not status["data_ready"]:
            phase = "warmup"
        elif status["ready_for_online"]:
            phase = "online"
        else:
            return "warmup_wait_online"
        self.episode_phase = phase
        return phase

    def finish_episode(self):
        self.episode_phase = None


class OnlineLearner:
    def __init__(self, learner, journal, config, *, seed=42, start=True):
        self.learner, self.journal, self.config = learner, journal, config
        self.condition = threading.Condition(threading.RLock())
        self.stopped = False
        self.error = None
        self.replay = ReplayBuffer(learner.spec, capacity=config.replay.capacity, seed=seed)
        records = journal.read_all() if journal is not None else []
        self.replay.extend(records)
        saved = learner.runtime_state
        self.adds_total = max(len(records), saved.get("adds_total", 0))
        self.ready_adds = saved.get("warmup_ready_adds_total")
        self.completed_steps = learner.steps
        self.actor = copy.deepcopy(learner.actor).eval().requires_grad_(requires_grad=False)
        self.actor_version = learner.steps // learner.config.actor_update_period - 1
        self.published_step = learner.steps if self.actor_version >= 0 else -1
        self.thread = None
        with self.condition:
            self._latch_data_ready()
        if config.mode == "eval" and self.actor_version < 0:
            raise ValueError("RLT eval requires a checkpoint with at least one actor update")
        if config.mode == "online_rl" and not config.collect.enabled and not self.status()["data_ready"]:
            raise ValueError("Collection disabled: provide a replay containing warmup.min_transitions records")
        if start and config.mode == "online_rl":
            self.start()

    def _latch_data_ready(self):
        if self.ready_adds is None and len(self.replay) >= self.config.warmup.min_transitions:
            self.ready_adds = self.adds_total
            logger.info("Warmup data ready: transitions=%s updates=%s", len(self.replay), self.config.warmup.updates)

    def _status(self):
        data_ready = len(self.replay) >= self.config.warmup.min_transitions
        training_ready = data_ready and self.completed_steps >= self.config.warmup.updates
        ready = training_ready and self.published_step >= self.config.warmup.updates and self.actor_version >= 0
        return {
            "replay_size": len(self.replay),
            "adds_total": self.adds_total,
            "global_step": self.completed_steps,
            "warmup_required_updates": self.config.warmup.updates,
            "actor_version": self.actor_version,
            "published_step": self.published_step,
            "data_ready": data_ready,
            "ready_for_online": ready and self.error is None,
            "learner_error": self.error,
        }

    def status(self):
        with self.condition:
            return self._status()

    def start(self):
        if self.config.mode != "online_rl":
            raise ValueError("Eval mode cannot start a learner")
        if self.thread is not None:
            raise ValueError("Learner already started")
        self.thread = threading.Thread(target=self._run, name="rlt-learner", daemon=True)
        self.thread.start()

    def commit_episode(self, records, metadata):
        if self.journal is None:
            raise ValueError("Collection is disabled")
        with self.condition:
            episode_id = records[0].episode_id
            if self.journal.episode_metadata(episode_id) is not None:
                raise ValueError("Episode ID already committed; use a unique episode_id")
            self.journal.replace_episode(records, metadata=metadata)
            self.replay.extend(records)
            self.adds_total += len(records)
            self._latch_data_ready()
            self.condition.notify_all()

    def _desired_updates(self):
        if self.ready_adds is None or len(self.replay) < self.config.warmup.min_transitions:
            return self.completed_steps
        if self.config.learner.freeze_after_warmup:
            return self.config.warmup.updates
        online_adds = max(self.adds_total - self.ready_adds, 0)
        return self.config.warmup.updates + online_adds * self.config.learner.grad_updates_per_cycle

    def train_once(self):
        """Used by the worker; also permits deterministic CPU verification."""
        with self.condition:
            if self.config.mode != "online_rl" or self.stopped or self.completed_steps >= self._desired_updates():
                return False
            batch = self.replay.sample(self.config.learner.batch_size)
            warmup = self.completed_steps < self.config.warmup.updates
        self.learner.config = self.config.learner.training_config(warmup=warmup)
        metrics = self.learner.update(batch)
        with self.condition:
            self.completed_steps = self.learner.steps
            # Always publish at the readiness boundary, even between intervals.
            if (
                self.completed_steps % self.config.learner.publish_interval == 0
                or self.completed_steps >= self.config.warmup.updates > self.published_step
            ):
                self._publish()
            self.condition.notify_all()
        if self.completed_steps % self.config.learner.checkpoint_interval == 0:
            self.save()
        if self.completed_steps == 1 or self.completed_steps % 100 == 0:
            logger.info("RLT update=%s metrics=%s", self.completed_steps, metrics)
        return True

    def _publish(self):
        self.actor.load_state_dict(self.learner.actor.state_dict(), strict=True)
        self.actor_version = self.completed_steps // self.learner.config.actor_update_period - 1
        self.published_step = self.completed_steps

    def propose(self, feature):
        proprio = np.asarray(feature["proprio"], dtype=np.float32)[None]
        ref = self.learner.adapter.normalize(np.asarray(feature["ref_chunk"])[None], proprio)
        device = self.learner.device
        with self.condition, torch.no_grad():
            if self.actor_version < 0:
                raise ValueError("No trained actor snapshot available")
            inputs = [
                torch.as_tensor(value, dtype=torch.float32, device=device)
                for value in (np.asarray(feature["z_rlt"])[None], proprio, ref)
            ]
            grip = (
                torch.as_tensor(feature["z_grip"][None], dtype=torch.float32, device=device)
                if self.learner.spec.grip_dim
                else None
            )
            proposal = self.actor.sample(*inputs, grip, deterministic=self.config.deterministic).cpu().numpy()
            version = self.actor_version
        return self.learner.adapter.denormalize(proposal, proprio)[0], version

    def save(self):
        with self.condition:
            self.learner.runtime_state.update(
                adds_total=self.adds_total,
                warmup_ready_adds_total=self.ready_adds,
                warmup_updates=self.config.warmup.updates,
            )
        self.learner.save(self.config.learner.checkpoint_path)

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or self.completed_steps < self._desired_updates())
                    if self.stopped:
                        break
                self.train_once()
            self.save()
        except Exception as error:
            logger.exception("RLT learner stopped")
            with self.condition:
                self.error = str(error)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join()
