"""Robot-side episode and execution feedback for openpi-rlt-online-v1."""

import time
from typing import Dict, Optional

from openpi_client import base_policy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class RLTClientPolicy(base_policy.BasePolicy):
    def __init__(self, host="0.0.0.0", port=8001, *, transport=None):
        self.transport = transport if transport is not None else WebsocketClientPolicy(host, port)
        if self.transport.get_server_metadata().get("rlt_online_protocol") != "openpi-rlt-online-v1":
            raise ValueError("Server does not implement openpi-rlt-online-v1")
        self.episode_id = None
        self.pending_step = None

    def get_server_metadata(self) -> Dict:
        return self.transport.get_server_metadata()

    def _request(self, event, **values):
        # Episode mutations and execution feedback must never be replayed on reconnect.
        return self.transport.infer({"rlt_event": event, **values}, retry_on_disconnect=False)

    def status(self) -> Dict:
        return self._request("status")

    def begin_episode(self, episode_id: str, *, policy_mode="actor", wait=True, timeout=300.0) -> Dict:
        if self.episode_id is not None:
            raise ValueError("Finish or abort the current episode first")
        deadline = time.monotonic() + timeout
        while True:
            result = self._request("begin_episode", episode_id=episode_id, policy_mode=policy_mode)
            if result["accepted"]:
                self.episode_id = episode_id
                return result
            if not wait:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Warmup still running; inspect status and retry before starting the robot")
            time.sleep(0.25)

    def infer(self, obs: Optional[Dict] = None, *, in_critical_phase: Optional[bool] = None) -> Dict:
        values = {"episode_id": self.episode_id}
        if obs is not None:
            values["observation"] = obs
        if in_critical_phase is not None:
            values["in_critical_phase"] = in_critical_phase
        result = self._request("infer", **values)
        self.pending_step = result["step_id"]
        return result

    def observe(self, executed_actions, next_observation: Dict, *, source=None) -> Dict:
        values = {
            "episode_id": self.episode_id,
            "step_id": self.pending_step,
            "executed_actions": executed_actions,
            "next_observation": next_observation,
        }
        if source is not None:
            values["source"] = source
        result = self._request("observe", **values)
        self.pending_step = None
        return result

    def end_episode(self, success: bool) -> Dict:
        result = self._request("end_episode", episode_id=self.episode_id, success=success)
        self.episode_id, self.pending_step = None, None
        return result

    def reset(self) -> None:
        if self.episode_id is not None:
            self._request("abort_episode", episode_id=self.episode_id)
        self.episode_id, self.pending_step = None, None

    def close(self) -> None:
        self.transport.close()
