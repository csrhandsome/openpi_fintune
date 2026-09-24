"""WebSocket client for pushing VLA predictions to Reactive Desk."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import websockets
import websockets.sync.client

logger = logging.getLogger(__name__)


class ReactiveDeskVlaClient:
    """Push FK-projected VLA predictions to Reactive Desk's ingest WebSocket."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/ws/VlaIngest",
        *,
        enabled: bool = True,
        connect_on_init: bool = False,
    ) -> None:
        self._uri = _build_ws_uri(host, port, path)
        self._enabled = enabled
        self._ws: websockets.sync.client.ClientConnection | None = None

        if self._enabled and connect_on_init:
            self.connect()

    def connect(self) -> None:
        if not self._enabled or self._ws is not None:
            return

        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
        )
        # Reactive Desk sends a connection message after accepting the socket.
        self._ws.recv()
        logger.info("Connected to Reactive Desk VLA ingest at %s", self._uri)

    def send_predictions(
        self,
        xyz: np.ndarray,
        probabilities: np.ndarray,
        *,
        prompt: str = "",
        is_executing: bool = True,
    ) -> bool:
        if not self._enabled:
            return False

        xyz = np.asarray(xyz, dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if xyz.ndim == 1:
            xyz = xyz.reshape(1, -1)

        predictions = [
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]) if point.shape[0] > 2 else 0.0,
                "probability": float(probabilities[rank]) if rank < probabilities.shape[0] else 1.0,
                "rank": int(rank),
            }
            for rank, point in enumerate(xyz)
        ]
        payload = {
            "type": "vla_predictions",
            "predictions": predictions,
            "is_executing": is_executing,
            "current_prompt": prompt,
        }

        try:
            self.connect()
            if self._ws is None:
                return False
            self._ws.send(json.dumps(payload))
            # The current Reactive Desk ingest endpoint replies with an ack.
            self._ws.recv()
            return True
        except websockets.ConnectionClosed:
            self._ws = None
            logger.warning("Reactive Desk VLA ingest connection closed")
            return False
        except OSError as exc:
            self._ws = None
            logger.warning("Reactive Desk VLA ingest send failed: %s", exc)
            return False

    def close(self) -> None:
        if self._ws is None:
            return
        self._ws.close()
        self._ws = None

    def __enter__(self) -> ReactiveDeskVlaClient:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _build_ws_uri(host: str, port: int, path: str) -> str:
    uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"

    if path:
        path = path if path.startswith("/") else f"/{path}"
        if not uri.endswith(path):
            uri = f"{uri.rstrip('/')}{path}"

    return uri
