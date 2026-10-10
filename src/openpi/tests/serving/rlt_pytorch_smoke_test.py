"""Exercise YAML dispatch, frozen readout training and WebSocket serving."""

import asyncio
import dataclasses
import json
import sys
from types import SimpleNamespace

import numpy as np
from openpi_client import msgpack_numpy
import pytest
import torch
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as websocket_serve
import yaml

from openpi.models import pi0_config
from openpi.models_pytorch.rlinf.model import Observation
from openpi.policies.policy import Policy
from openpi.rlt.force_rlt import serve as force_rlt
from openpi.rlt.force_rlt.learner import ChunkLearner
from openpi.rlt.force_rlt.learner import LearnerConfig
from openpi.rlt.force_rlt.networks import NetworkConfig
from openpi.serving import runner as server
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as train_config
from openpi.training import rlt
from scripts import train


class FakeVLA(torch.nn.Module):
    action_horizon = 4
    action_dim = 32

    def __init__(self):
        super().__init__()
        self.prefix = torch.nn.Parameter(torch.ones(8))
        self.llm = SimpleNamespace(configs=[SimpleNamespace(width=8)])

    def extract_prefix_embeddings(self, observation, *, image_only, train=False):
        assert image_only
        assert not torch.is_grad_enabled()
        batch = observation.state.shape[0]
        prefix = self.prefix[None, None].expand(batch, 3, 8)
        mask = torch.tensor([[True, True, False]], device=prefix.device).expand(batch, -1)
        return prefix, mask

    def sample_actions_with_prefix(self, observation, **kwargs):
        del kwargs
        prefix, mask = self.extract_prefix_embeddings(observation, image_only=True)
        actions = torch.arange(128, dtype=torch.float32, device=prefix.device).reshape(1, 4, 32)
        return actions, prefix, mask

    def sample_actions(self, observation, **kwargs):
        return self.sample_actions_with_prefix(observation, **kwargs)[0]


@pytest.mark.parametrize(
    ("group", "rgb", "action_space"),
    [
        ("B", False, "joint"),
        ("C", False, "joint"),
        ("C", True, "joint"),
        (None, False, "ee_delta"),
    ],
)
def test_yaml_train_and_server_roundtrip(tmp_path, monkeypatch, group, rgb, action_space):
    checkpoint = tmp_path / "vla"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"fixture checkpoint")
    readout_dir = tmp_path / "readout"
    settings = train_config.get_config("openpi_rlt")
    settings = dataclasses.replace(
        settings,
        name="fixture",
        num_train_steps=1,
        save_interval=1,
        model=dataclasses.replace(settings.model, action_horizon=4),
        data=dataclasses.replace(settings.data, action_space=action_space, action_dim=8 if group else 7),
        rlt=dataclasses.replace(
            settings.rlt,
            vla_checkpoint=str(checkpoint),
            output_dir=str(readout_dir),
            embed_dim=8,
            num_layers=1,
            device="cpu",
        ),
    )
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(yaml.safe_dump(train_config.to_dict(settings)))
    settings = train_config.get_config(yaml_path)
    model = FakeVLA()
    monkeypatch.setattr(pi0_config.Pi0Config, "load_pytorch", lambda *a: model)
    monkeypatch.setattr(rlt, "validate_dataset", lambda *a: None)

    def fake_data(self, *args):
        assert self.assets.assets_dir == str(checkpoint / "assets")
        return SimpleNamespace(norm_stats={"state": object()})

    monkeypatch.setattr(train_config.LeRobotFrankaDataConfig, "create", fake_data)
    observation = Observation(images={}, image_masks={}, state=torch.zeros(1, 32))
    monkeypatch.setattr(rlt.data_loader, "create_data_loader", lambda *a, **kw: [(observation, None)])
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(yaml_path)])
    original_train = rlt.train
    monkeypatch.setattr(rlt, "train", lambda config: original_train(config, defaults=rlt.ReadoutDefaults(alpha=0)))
    train.main()
    assert not model.prefix.requires_grad
    torch.testing.assert_close(model.prefix, torch.ones(8))
    manifest = json.loads((readout_dir / "manifest.json").read_text())
    assert manifest["prefix_shape"] == [3, 8]
    assert manifest["physical_action_dim"] == (8 if group else 7)
    assert manifest["action_representation"] == action_space
    snapshot = train_config.get_config(readout_dir / "config.yaml")
    assert snapshot.batch_size == settings.batch_size
    assert snapshot.rlt.vla_checkpoint == str(checkpoint)
    assert (readout_dir / "1/readout.safetensors").is_file()

    def fake_policy(*args, **kwargs):
        assert args[1] == readout_dir / "1"
        return Policy(
            FakeVLA(),
            transforms=(),
            output_transforms=(lambda data: {"actions": data["actions"][:, : 8 if group else 7]},),
            pytorch_device="cpu",
            is_pytorch=True,
            metadata=dict(settings.policy_metadata),
        )

    request = {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.ones((), dtype=bool)},
        "state": np.zeros(8 if group else 7, np.float32),
        "observation.joint_position": np.zeros(7, np.float32),
        "monotonic_ts": np.array(10.0),
        "O_T_TCP": np.eye(4, dtype=np.float32),
    }
    monkeypatch.setattr(server.policy_config, "create_trained_policy", fake_policy)
    if group is None:
        policy = server.create_policy(server.Args(config=str(yaml_path), device="cpu"))
        # Standalone token inference does not require the contact actor's TCP anchor.
        request.pop("monotonic_ts")
        request.pop("O_T_TCP")
        assert "ref_q" not in policy.infer(request)
        metadata = policy.metadata
        assert metadata["rlt_readout_protocol"] == "pytorch-rl-token-v1"
        assert "rlt_protocol" not in metadata
    else:
        network = NetworkConfig(z_rlt_dim=8, z_grip_dim=24 if rgb else 12, horizon=4, executed_steps=2, hidden_dim=16)
        actor_checkpoint = tmp_path / f"{group}.pt"
        ChunkLearner(LearnerConfig(network, zero_grip=group == "B")).save(actor_checkpoint)
        if group == "C":
            request.update(z_grip=np.ones(12, np.float32), finger_valid_mask=np.ones(4, bool))
        if rgb:

            class TinyEncoder:
                feature_dim = 1

                def encode_pairs(self, left, right):
                    return np.stack([left.mean(axis=(1, 2, 3)), right.mean(axis=(1, 2, 3))], axis=1)[..., None]

            monkeypatch.setattr(force_rlt, "SharedFingerEncoder", lambda *a, **kw: TinyEncoder())
            request.pop("z_grip")
            request.update(
                {
                    "observation.gripper_image_left": np.ones((4, 2, 2, 3), np.uint8),
                    "observation.gripper_image_right": np.ones((4, 2, 2, 3), np.uint8),
                    "finger_timestamps": np.repeat(np.array([9.9, 9.93, 9.96, 9.99])[:, None], 2, axis=1),
                    "finger_baseline_left": np.zeros((2, 2, 3), np.uint8),
                    "finger_baseline_right": np.zeros((2, 2, 3), np.uint8),
                    "finger_session_id": "smoke-rgb",
                }
            )
        received = {}

        class FakeServer:
            def __init__(self, policy, *, host, port, metadata):
                assert host == "0.0.0.0"
                assert port == 8002
                received.update(policy=policy, metadata=metadata)

            def serve_forever(self):
                received["policy"].infer(request)

        monkeypatch.setattr(force_rlt, "WebsocketPolicyServer", FakeServer)
        args = [
            "force_rlt",
            "--config",
            str(yaml_path),
            "--group",
            group,
            "--actor-checkpoint",
            str(actor_checkpoint),
            "--vla-checkpoint",
            str(checkpoint),
            "--readout",
            str(readout_dir),
            "--device",
            "cpu",
        ]
        if rgb:
            args += ["--finger-weights", str(tmp_path / "weights.pt")]
        monkeypatch.setattr(sys, "argv", args)
        force_rlt.main()
        policy, metadata = received["policy"], received["metadata"]

    response = policy.infer(request)
    assert response["z_rlt"].shape == (8,)
    assert response["ref_chunk"].shape == (4, 8 if group else 7)
    if group:
        assert response["proposed_q"].shape == (4, 7)
        np.testing.assert_array_equal(response["ref_q"], response["actions"][:, :7])
    if group == "C":
        with pytest.raises(ValueError, match="valid finger history"):
            policy.infer({**request, "finger_valid_mask": np.array([True, False])})

    async def exchange():
        websocket_server = WebsocketPolicyServer(policy, host="127.0.0.1", metadata=metadata)
        async with websocket_serve(websocket_server._handler, "127.0.0.1", 0) as listener:  # noqa: SLF001
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as websocket:
                wire_metadata = msgpack_numpy.unpackb(await websocket.recv())
                await websocket.send(msgpack_numpy.Packer().pack(request))
                return wire_metadata, msgpack_numpy.unpackb(await websocket.recv())

    wire_metadata, wire_response = asyncio.run(exchange())
    protocol_key = "rlt_protocol" if group else "rlt_readout_protocol"
    assert wire_metadata[protocol_key] == manifest["format"]
    np.testing.assert_array_equal(wire_response["ref_chunk"], response["ref_chunk"])
