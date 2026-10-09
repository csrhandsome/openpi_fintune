"""Exercise the readout train/serve entry points without a hardware checkpoint."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest
import torch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as websocket_serve

from openpi.force_rlt.learner import ChunkLearner
from openpi.force_rlt.learner import LearnerConfig
from openpi.force_rlt.networks import NetworkConfig
from openpi.models_pytorch.rlinf.model import Observation
from openpi.policies.policy import Policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def _script(name: str):
    """Load an entry-point script from the parent ``scripts/`` directory."""
    path = Path(__file__).resolve().parent.parent / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class FakeAssets:
    assets_dir: str | None = None


@dataclasses.dataclass(frozen=True)
class FakeData:
    assets: FakeAssets = dataclasses.field(default_factory=FakeAssets)

    def create(self, assets_dirs, model):
        del assets_dirs, model
        assert self.assets.assets_dir is not None
        assert Path(self.assets.assets_dir).is_dir()
        return SimpleNamespace(norm_stats={"state": object()})


class FakeVLA(torch.nn.Module):
    action_horizon = 4
    action_dim = 32

    def __init__(self):
        super().__init__()
        self.llm = SimpleNamespace(configs=[SimpleNamespace(width=8)])

    def extract_prefix_embeddings(self, observation, *, image_only):
        assert image_only
        batch = observation.state.shape[0]
        prefix = torch.ones(batch, 3, 8, device=observation.state.device)
        mask = torch.tensor([[True, True, False]], device=prefix.device).expand(batch, -1)
        return prefix, mask

    def sample_actions_with_prefix(self, observation, **kwargs):
        del kwargs
        prefix, mask = self.extract_prefix_embeddings(observation, image_only=True)
        actions = torch.arange(128, dtype=torch.float32, device=prefix.device).reshape(1, 4, 32)
        return actions, prefix, mask

    def sample_actions(self, observation, **kwargs):
        return self.sample_actions_with_prefix(observation, **kwargs)[0]


@dataclasses.dataclass(frozen=True)
class FakeEEData(FakeData):
    pass


@dataclasses.dataclass(frozen=True)
class FakeConfig:
    data: FakeData = dataclasses.field(default_factory=FakeData)
    model: object = None
    assets_dirs: Path = dataclasses.field(default_factory=lambda: Path("unused"))
    policy_metadata: dict = dataclasses.field(default_factory=lambda: {"action_space": "joint"})



@pytest.mark.parametrize(
    ("group", "rgb", "action_space"),
    [("B", False, "joint"), ("C", False, "joint"), ("C", True, "joint"), ("B", False, "ee")],
)
def test_merged_server_runs_vla_readout_and_actor_in_one_request(tmp_path, monkeypatch, group, rgb, action_space):
    train = _script("train_rlt_pytorch.py")
    serve = _script("serve_force_rlt_pytorch.py")
    config = FakeConfig(
        data=FakeEEData() if action_space == "ee" else FakeData(),
        model=SimpleNamespace(load_pytorch=lambda *_: FakeVLA(), action_horizon=4, action_dim=32),
        policy_metadata={"action_space": action_space},
    )
    config_name = "pi0_rlt" if action_space == "ee" else "fixture"
    checkpoint = tmp_path / "vla"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "assets" / "fixture.txt").write_text("stats", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fixture checkpoint")
    readout_dir = tmp_path / "readout"
    monkeypatch.setattr(train.train_config, "get_config", lambda name: config)
    monkeypatch.setattr(train.train_config, "LeRobotFrankaJointDataConfig", FakeData)
    monkeypatch.setattr(train.train_config, "LeRobotFrankaEEDataConfig", FakeEEData)
    observation = Observation(images={}, image_masks={}, state=torch.zeros(1, 32))
    monkeypatch.setattr(train.data_loader, "create_data_loader", lambda *a, **kw: [(observation, None)])
    monkeypatch.setattr(
        sys, "argv",
        ["train_rlt_pytorch.py", "--config", config_name, "--vla-checkpoint", str(checkpoint),
         "--output", str(readout_dir), "--steps", "1", "--embed-dim", "8",
         "--num-layers", "1", "--device", "cpu"],
    )
    train.main()
    manifest = json.loads((readout_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["prefix_shape"] == [3, 8]
    assert manifest["action_space"] == action_space
    assert manifest["physical_action_dim"] == (7 if action_space == "ee" else 8)

    if action_space == "ee":
        assert "physical_joint_dim" not in manifest
        monkeypatch.setattr(sys, "argv", [
            "serve_force_rlt_pytorch.py", "--config", config_name, "--group", group,
            "--actor-checkpoint", str(tmp_path / "unused.pt"), "--vla-checkpoint", str(checkpoint),
            "--readout", str(readout_dir), "--device", "cpu",
        ])
        with pytest.raises(ValueError, match="EE-to-joint adapter"):
            serve.main()
        return

    network = NetworkConfig(z_rlt_dim=8, z_grip_dim=24 if rgb else 12, horizon=4, executed_steps=2, hidden_dim=16)
    actor_checkpoint = tmp_path / f"{group}.pt"
    ChunkLearner(LearnerConfig(network, zero_grip=group == "B")).save(actor_checkpoint)

    def fake_policy(*args, **kwargs):
        del args, kwargs
        return Policy(
            FakeVLA(),
            transforms=(),
            output_transforms=(lambda data: {"actions": data["actions"][:, :8]},),
            pytorch_device="cpu",
            is_pytorch=True,
            metadata={"action_fps": 30.0, "action_space": "joint", "action_horizon": 4, "action_dim": 8},
        )

    request = {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        "state": np.zeros(8, np.float32),
        "observation.joint_position": np.zeros(7, np.float32),
        "monotonic_ts": np.array(10.0),
        "O_T_TCP": np.eye(4, dtype=np.float32),
    }
    if group == "C":
        request.update(z_grip=np.ones(12, np.float32), finger_valid_mask=np.ones(4, bool))
    if rgb:
        class TinyEncoder:
            feature_dim = 1

            def encode_pairs(self, left, right):
                return np.stack([left.mean(axis=(1, 2, 3)), right.mean(axis=(1, 2, 3))], axis=1)[..., None]

        monkeypatch.setattr(serve, "SharedFingerEncoder", lambda *a, **kw: TinyEncoder())
        request.pop("z_grip")
        request.update({
            "observation.gripper_image_left": np.ones((4, 2, 2, 3), np.uint8),
            "observation.gripper_image_right": np.ones((4, 2, 2, 3), np.uint8),
            "finger_timestamps": np.repeat(np.array([9.9, 9.93, 9.96, 9.99])[:, None], 2, axis=1),
            "finger_baseline_left": np.zeros((2, 2, 3), np.uint8),
            "finger_baseline_right": np.zeros((2, 2, 3), np.uint8),
            "finger_session_id": "smoke-rgb",
        })
    received = {}

    class FakeServer:
        def __init__(self, policy, *, host, port, metadata):
            assert host == "0.0.0.0"
            assert port == 8002
            received["policy"] = policy
            received["metadata"] = metadata

        def serve_forever(self):
            received["response"] = received["policy"].infer(request)

    monkeypatch.setattr(serve.train_config, "get_config", lambda name: config)
    monkeypatch.setattr(serve.policy_config, "create_trained_policy", fake_policy)
    monkeypatch.setattr(serve, "WebsocketPolicyServer", FakeServer)
    args = [
        "serve_force_rlt_pytorch.py", "--config", "fixture", "--group", group,
        "--actor-checkpoint", str(actor_checkpoint), "--vla-checkpoint", str(checkpoint),
        "--readout", str(readout_dir), "--device", "cpu",
    ]
    if rgb:
        args += ["--finger-weights", str(tmp_path / "weights.pt")]
    monkeypatch.setattr(sys, "argv", args)
    serve.main()

    metadata, response = received["metadata"], received["response"]
    assert metadata["rlt_protocol"] == "pytorch-rl-token-v1"
    assert metadata["rlt_actor_protocol"] == "pytorch-force-rlt-actor-v1"
    assert metadata["group"] == group
    if rgb:
        assert metadata["finger_input_protocol"] == "panda-finger-rgb-history-v1"
    assert response["z_rlt"].shape == (8,)
    assert response["proposed_q"].shape == (4, 7)
    np.testing.assert_array_equal(response["ref_q"], response["actions"][:, :7])
    assert np.isfinite(response["proposed_q"]).all()
    if group == "C":
        with pytest.raises(ValueError, match="valid finger history"):
            received["policy"].infer({**request, "finger_valid_mask": np.array([True, False])})
    else:
        np.testing.assert_array_equal(
            response["proposed_q"],
            received["policy"].infer({**request, "z_grip": np.ones(12, np.float32)})["proposed_q"],
        )

    async def exchange():
        server = WebsocketPolicyServer(received["policy"], host="127.0.0.1", metadata=metadata)
        async with websocket_serve(server._handler, "127.0.0.1", 0) as listener:  # noqa: SLF001
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as websocket:
                wire_metadata = msgpack_numpy.unpackb(await websocket.recv())
                await websocket.send(msgpack_numpy.Packer().pack(request))
                wire_response = msgpack_numpy.unpackb(await websocket.recv())
                return wire_metadata, wire_response

    wire_metadata, wire_response = asyncio.run(exchange())
    assert wire_metadata["rlt_protocol"] == manifest["format"]
    assert not any("sha256" in key for key in manifest | wire_metadata)
    np.testing.assert_array_equal(wire_response["ref_q"], response["ref_q"])
    np.testing.assert_array_equal(wire_response["proposed_q"], response["proposed_q"])
