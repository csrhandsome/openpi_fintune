import concurrent.futures
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Returns one action at a time while prefetching the next RTC-conditioned chunk.

    When ``execution_horizon`` actions from the current chunk have been returned,
    the broker starts a background request for the next chunk and keeps returning
    the current chunk's overlapping prefix actions. It switches to the new chunk
    after ``inference_delay`` prefix actions have been consumed, blocking only if
    the background request has not finished by then.
    """

    def __init__(self, policy: _base_policy.BasePolicy, execution_horizon: int, inference_delay: int):
        if execution_horizon <= 0:
            raise ValueError("execution_horizon must be > 0")
        if inference_delay < 0:
            raise ValueError("inference_delay must be >= 0")
        self._policy = policy
        self._execution_horizon = int(execution_horizon)
        self._inference_delay = int(inference_delay)
        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._pending_future: concurrent.futures.Future[Dict[str, np.ndarray]] | None = None
        self._pending_switch_step = 0
        self._pending_delay = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-action-broker")

    def _build_request(self, obs: Dict) -> tuple[Dict, int]:  # noqa: UP006
        request = dict(obs)
        rtc_delay = 0
        if self._last_results is not None and self._inference_delay > 0:
            previous_actions = np.asarray(self._last_results["actions"], dtype=np.float32)
            prefix_start = self._cur_step
            prefix_end = prefix_start + self._inference_delay
            action_prefix = previous_actions[prefix_start:prefix_end]
            rtc_delay = min(self._inference_delay, action_prefix.shape[0])
            if rtc_delay > 0:
                request["action_prefix"] = action_prefix[:rtc_delay]
                request["rtc_delay"] = np.asarray(rtc_delay, dtype=np.int32)
        return request, int(rtc_delay)

    def _start_prefetch(self, obs: Dict) -> None:  # noqa: UP006
        request, rtc_delay = self._build_request(obs)
        self._pending_future = self._executor.submit(self._policy.infer, request)
        self._pending_delay = rtc_delay
        self._pending_switch_step = self._cur_step + rtc_delay

    def _adopt_pending(self) -> None:
        if self._pending_future is None:
            return
        self._last_results = self._pending_future.result()
        self._cur_step = self._pending_delay
        self._pending_future = None
        self._pending_switch_step = 0
        self._pending_delay = 0

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        if self._pending_future is None and self._cur_step >= self._execution_horizon:
            self._start_prefetch(obs)

        if self._pending_future is not None:
            action_count = np.asarray(self._last_results["actions"]).shape[0]
            switch_step = min(self._pending_switch_step, action_count)
            if self._cur_step >= switch_step:
                self._adopt_pending()

        def slicer(x):
            if isinstance(x, np.ndarray):
                index = min(self._cur_step, x.shape[0] - 1)
                return x[index, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._last_results = None
        self._pending_future = None
        self._pending_switch_step = 0
        self._pending_delay = 0
        self._cur_step = 0

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
