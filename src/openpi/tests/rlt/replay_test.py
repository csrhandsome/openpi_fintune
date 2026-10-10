"""Check sparse rewards, terminal windows, reannotation and real LeRobot import."""

import json
import sys
from types import SimpleNamespace

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest

from openpi.rlt import build_replay
from openpi.rlt.dataset import episode_actions
from openpi.rlt.dataset import import_dataset
from openpi.rlt.dataset import read_annotation
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.episode import build_episode_transitions
from openpi.rlt.features import FeatureCache
from openpi.rlt.replay import ReplayBuffer
from openpi.rlt.replay import ReplayJournal
from openpi.rlt.rewards import TerminalReward


def episode(spec, length=7, *, success=True):
    features = [
        {
            "z_rlt": np.full(spec.z_dim, index, np.float32),
            "proprio": np.zeros(spec.proprio_dim, np.float32),
            "ref_chunk": np.full((spec.horizon, spec.action_dim), 0.2, np.float32),
        }
        for index in range(length + 1)
    ]
    actions = np.ones((length, spec.action_dim), np.float32)
    return build_episode_transitions(
        "episode", features, actions, ["BASE"] * length, spec=spec, success=success, stride=3
    )


def test_terminal_reward_and_windows():
    spec = ReplaySpec(4, 7, 4, 3, 7)
    records = episode(spec, length=7)
    assert [r.step_id for r in records] == [0, 3, 4]
    assert [r.done for r in records] == [False, False, True]
    np.testing.assert_array_equal(records[-1].rewards, [0, 0, 1])
    assert sum(r.rewards.sum() for r in records) == 1
    assert all(np.all(r.rewards == 0) for r in episode(spec, success=False))
    with pytest.raises(ValueError, match="explicit"):
        TerminalReward().generate(3, success=None)
    np.testing.assert_allclose(TerminalReward(2, -1, 0.1).generate(3, success=False), [0.1, 0.1, -1])


def test_partial_window_masks_padding_and_requires_next_state():
    spec = ReplaySpec(4, 7, 4, 3, 7)
    record = episode(spec, length=1)[0]
    assert record.done
    np.testing.assert_array_equal(record.executed_mask, [True, False, False])
    np.testing.assert_array_equal(record.rewards, [1, 0, 0])
    assert record.next_z_rlt[0] == 1
    with pytest.raises(ValueError, match="one more state"):
        build_episode_transitions("bad", [], np.zeros((1, 7)), ["BASE"], spec=spec, success=True)


def test_pickle_reannotation_restore_and_buffer(tmp_path):
    spec = ReplaySpec(4, 7, 4, 3, 7)
    journal = ReplayJournal(tmp_path / "replay", spec)
    records = episode(spec)
    journal.replace_episode(records, metadata={"success": True})
    journal.replace_episode(episode(spec, success=False), metadata={"success": False})
    restored = ReplayJournal(tmp_path / "replay")
    assert len(restored) == len(records)
    assert len(list(restored.path.glob("episode_*.pkl"))) == 1
    assert restored.episode_metadata("episode")["success"] is False
    assert all(np.all(record.rewards == 0) for record in restored.read_all())
    buffer = ReplayBuffer(spec, capacity=2)
    buffer.extend(restored.read_all())
    assert len(buffer) == 2
    assert len(buffer.sample(8)) == 8
    with pytest.raises(ValueError, match="specification"):
        ReplayJournal(tmp_path / "replay", ReplaySpec(5, 7, 4, 3, 7))
    restored.remove_episode("episode")
    assert len(restored) == 0


def test_command_alignment_uses_past_target_not_measured_state(tmp_path):
    rows = [
        {
            "host_sample_monotonic_ns": stamp,
            "target_ee_position": [position, 0, 0],
            "target_ee_orientation_xyzw": [0, 0, 0, 1],
            "gripper_position": 0.4,
            "ee_position": [9, 9, 9],
            "vr_arm_enabled": False,
        }
        for stamp, position in ((1_000_000_000, 0.1), (1_030_000_000, 0.2), (1_034_000_000, 0.3))
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    actions, sources = episode_actions(
        tmp_path,
        {"action_trace": path.name},
        [{}, {}],
        np.array([1_000_000_000, 1_033_333_333]),
        space="ee",
        action_source="command",
    )
    assert actions[0, 0] == pytest.approx(0.2)
    assert actions[0, -1] == pytest.approx(0.4)
    assert sources == ["BASE"]
    with pytest.raises(ValueError, match="target_joint_position"):
        episode_actions(
            tmp_path,
            {"action_trace": path.name},
            [{}, {}],
            np.array([1_000_000_000, 1_033_333_333]),
            space="joint",
            action_source="command",
        )


def test_real_v3_import_frontend_labels_cache_and_reannotation(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    features = {
        key: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]}
        for key in ("observation.exterior_image", "observation.wrist_image_left")
    }
    features.update(
        {
            key: {"dtype": "float32", "shape": (dim,), "names": None}
            for key, dim in {
                "observation.ee_pose": 6,
                "observation.gripper_position": 1,
                "action.ee_pose": 6,
                "action.gripper_position": 1,
            }.items()
        }
    )
    writer = LeRobotDataset.create("local/dataset", root=root, fps=30, features=features, robot_type="panda")
    for ep in range(2):
        for i in range(6):
            writer.add_frame(
                {
                    "observation.exterior_image": np.full((8, 8, 3), i, np.uint8),
                    "observation.wrist_image_left": np.full((8, 8, 3), i + 1, np.uint8),
                    "observation.ee_pose": np.full(6, i / 100, np.float32),
                    "observation.gripper_position": np.ones(1, np.float32),
                    "action.ee_pose": np.full(6, (i + 1) / 100, np.float32),
                    "action.gripper_position": np.ones(1, np.float32),
                    "task": f"task {ep}",
                }
            )
        writer.save_episode()
    writer.finalize()
    sync = root / "episode_000000.sync.json"
    sync.write_text(json.dumps({"episode_index": 0, "success": True, "source": "inference"}))
    dataset = LeRobotDataset("local/dataset", root=root)
    calls = []

    def infer(observation):
        assert observation["prompt"] == "task 0"
        calls.append(observation)
        return {"z_rlt": np.ones(4, np.float32), "ref_chunk": np.zeros((4, 7), np.float32)}

    spec = ReplaySpec(4, 7, 4, 3, 7)
    cache = FeatureCache(
        tmp_path / "features", SimpleNamespace(infer=infer), model_id="fixture", horizon=4, action_dim=7
    )
    journal = ReplayJournal(tmp_path / "replay", spec)
    result = import_dataset(dataset, root, cache, journal, action_source="measured")
    assert result == {"imported_episodes": 1, "skipped_unannotated": 1, "transitions": 2}
    assert len(calls) == 4
    records = journal.read_all()
    assert records[-1].rewards[-1] == 1
    assert records[0].action_chunk[0, 0] == pytest.approx(0.01)
    assert records[0].proprio[0] == 0
    audio = root / "audio"
    audio.mkdir()
    sync.rename(audio / sync.name)
    sync = audio / sync.name
    sync.write_text(json.dumps({"episode_index": 0, "success": False, "source": "inference"}))
    assert read_annotation(root, 0)["success"] is False
    import_dataset(dataset, root, cache, journal, action_source="measured")
    assert len(calls) == 4
    assert len(journal) == 2
    assert all(np.all(record.rewards == 0) for record in journal.read_all())
    # Exercise the public construction command with real saved LeRobot rows and a tiny feature provider.
    from openpi.serving import runner

    checkpoint, readout = tmp_path / "vla", tmp_path / "readout"
    checkpoint.mkdir()
    readout.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"fixture VLA identity")
    (readout / "readout.safetensors").write_bytes(b"fixture readout identity")
    (readout / "manifest.json").write_text("{}")
    metadata = {
        "rlt_protocol": "pytorch-rl-token-v1",
        "z_rlt_dim": 4,
        "action_dim": 7,
        "action_horizon": 4,
        "action_space": "ee",
        "action_fps": 30.0,
    }
    monkeypatch.setattr(runner, "create_policy", lambda args: SimpleNamespace(infer=infer, metadata=metadata))
    cli_output = tmp_path / "cli_replay"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_replay",
            "--dataset",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--readout",
            str(readout),
            "--output",
            str(cli_output),
            "--chunk-len",
            "3",
            "--action-source",
            "measured",
        ],
    )
    build_replay.main()
    build_replay.main()
    assert len(calls) == 8
    cli_journal = ReplayJournal(cli_output)
    assert len(cli_journal) == 2
    assert all(not record.success for record in cli_journal.read_all())
    cache_other = FeatureCache(cache.path, SimpleNamespace(infer=infer), model_id="other", horizon=4, action_dim=7)
    with pytest.raises(ValueError, match="model/action semantics"):
        import_dataset(dataset, root, cache_other, journal, action_source="measured")
    sync.write_text(json.dumps({"episode_index": 0, "success": None}))
    import_dataset(dataset, root, cache, journal, action_source="measured")
    assert len(journal) == 0
