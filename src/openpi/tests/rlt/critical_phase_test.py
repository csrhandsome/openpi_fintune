"""Critical replay starts after the marked boundary and ends at the real terminal state."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.rlt.dataset import import_dataset
from openpi.rlt.episode import ReplaySpec
from openpi.rlt.replay import ReplayJournal


@pytest.fixture
def setup_episode(tmp_path):
    stamps = [1_000_000_000 + round(index * 1e9 / 30) for index in range(8)]
    items = [
        {
            "timestamp": index / 30,
            "observation.ee_pose": np.full(6, index, np.float32),
            "observation.gripper_position": np.zeros(1, np.float32),
        }
        for index in range(8)
    ]
    dataset = SimpleNamespace(
        meta=SimpleNamespace(fps=30, episodes=[{"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 8}])
    )

    class Dataset:
        meta = dataset.meta

        def __getitem__(self, index):
            return items[index]

    rows = [
        {
            "host_sample_monotonic_ns": stamp + 1,
            "target_ee_position": [index, 0, 0],
            "target_ee_orientation_xyzw": [0, 0, 0, 1],
            "target_gripper_position": 0.6,
            "control_source": "HUMAN" if index == 4 else "RL",
            "intervention_flag": index == 4,
            "in_critical_phase": index >= 3,
        }
        for index, stamp in enumerate(stamps[:-1])
    ]
    (tmp_path / "actions.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    sync = {
        "episode_index": 0,
        "success": True,
        "action_trace": "actions.jsonl",
        "frame_records": [{"host_frame_monotonic_ns": stamp} for stamp in stamps],
        "critical_phase_annotation_version": 1,
        "critical_phase_start_monotonic_ns": stamps[2] + 5,
    }
    path = tmp_path / "episode_000000.sync.json"

    def write_sync():
        path.write_text(json.dumps(sync))

    write_sync()
    calls = []

    def get(observation, proprio):
        calls.append(float(proprio[0]))
        return {
            "z_rlt": np.full(4, proprio[0], np.float32),
            "proprio": proprio,
            "ref_chunk": np.zeros((4, 7), np.float32),
        }

    cache = SimpleNamespace(model_id="frozen", get=get)
    journal = ReplayJournal(tmp_path / "replay", ReplaySpec(4, 7, 4, 3, 7))
    return Dataset(), tmp_path, cache, journal, sync, write_sync, stamps, calls


def test_conservative_boundary_human_and_real_terminal(setup_episode):
    dataset, root, cache, journal, _, _, _, calls = setup_episode
    stats = import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    records = journal.read_all()
    assert [record.step_id for record in records] == [3, 4]
    assert [record.done for record in records] == [False, True]
    assert records[0].z_rlt[0] == 3
    assert records[-1].next_z_rlt[0] == 7
    assert all(np.min(record.action_chunk[:, 0]) >= 3 for record in records)
    assert records[0].source_chunk.tolist() == ["RL", "HUMAN", "RL"]
    assert records[0].intervention_flag
    assert records[0].rewards.sum() == 0
    np.testing.assert_array_equal(records[-1].rewards, [0, 0, 1])
    assert stats["excluded_noncritical_steps"] == 3
    assert stats["included_steps"] == 4
    metadata = journal.episode_metadata(records[0].episode_id)
    assert metadata["phase_scope"] == "critical_phase"
    assert metadata["first_included_step"] == 3
    assert metadata["original_executed_steps"] == 7
    assert metadata["executed_steps"] == 4
    assert sorted(calls) == [3, 4, 6, 7]


@pytest.mark.parametrize("marker", [None, "missing"])
def test_no_marker_removes_old_entries_and_never_extracts_features(setup_episode, marker):
    dataset, root, cache, journal, sync, write_sync, _, calls = setup_episode
    import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    if marker == "missing":
        sync.pop("critical_phase_start_monotonic_ns")
        sync.pop("critical_phase_annotation_version")
    else:
        sync["critical_phase_start_monotonic_ns"] = marker
    write_sync()
    calls.clear()
    stats = import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    assert len(journal) == 0
    assert calls == []
    assert stats["skipped_no_critical_phase"] == 1
    assert stats["excluded_noncritical_steps"] == 7


def test_reannotation_partial_tail_and_empty_phase(setup_episode):
    dataset, root, cache, journal, sync, write_sync, stamps, _ = setup_episode
    import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    sync["critical_phase_start_monotonic_ns"] = stamps[6]
    sync["success"] = False
    write_sync()
    stats = import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    records = journal.read_all()
    assert len(records) == 1
    assert records[0].step_id == 6
    assert records[0].done
    assert not records[0].success
    np.testing.assert_array_equal(records[0].executed_mask, [True, False, False])
    assert records[0].rewards.sum() == 0
    assert records[0].next_z_rlt[0] == 7
    assert stats["included_steps"] == 1
    sync["critical_phase_start_monotonic_ns"] = stamps[-1]
    write_sync()
    stats = import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    assert len(journal) == 0
    assert stats["skipped_empty_critical_phase"] == 1
    assert stats["excluded_noncritical_steps"] == 7


@pytest.mark.parametrize("marker", [-1, 2.5, True, "100"])
def test_rejects_invalid_markers(setup_episode, marker):
    dataset, root, cache, journal, sync, write_sync, _, _ = setup_episode
    sync["critical_phase_start_monotonic_ns"] = marker
    write_sync()
    with pytest.raises(ValueError, match="nonnegative integer"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)


def test_requires_known_version_and_absolute_frames(setup_episode):
    dataset, root, cache, journal, sync, write_sync, _, _ = setup_episode
    sync.pop("critical_phase_annotation_version")
    write_sync()
    with pytest.raises(ValueError, match="annotation_version"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    sync["critical_phase_annotation_version"] = 1
    sync.pop("frame_records")
    sync.pop("action_trace")
    write_sync()
    with pytest.raises(ValueError, match="absolute sync frame timestamps"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)


def test_phase_scope_bound_to_journal(setup_episode):
    dataset, root, cache, journal, *_ = setup_episode
    import_dataset(dataset, root, cache, journal)  # API remains backward compatible: full episode.
    assert journal.read_all()[0].step_id == 0
    with pytest.raises(ValueError, match="model/action semantics"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)


def test_legacy_provenance_cannot_silently_enter_critical_replay(setup_episode):
    dataset, root, cache, journal, *_ = setup_episode
    journal.bind_provenance({"model_id": "frozen", "action_source": "command"})
    with pytest.raises(ValueError, match="model/action semantics"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    # Explicit full-task mode preserves the historical provenance shared with online serving.
    import_dataset(dataset, root, cache, journal)
    assert journal.read_all()[0].step_id == 0


def test_noncritical_camera_gap_is_excluded_but_critical_gap_is_rejected(setup_episode):
    dataset, root, cache, journal, sync, write_sync, stamps, _ = setup_episode
    # Startup has a long gap before the marked phase. Commands in the retained suffix stay aligned.
    for frame, stamp in zip(sync["frame_records"][:3], stamps[:3], strict=True):
        frame["host_frame_monotonic_ns"] = stamp - 200_000_000
    write_sync()
    import_dataset(dataset, root, cache, journal, critical_phase_only=True)
    assert journal.read_all()[0].step_id == 3
    sync["frame_records"][4]["host_frame_monotonic_ns"] -= 20_000_000
    write_sync()
    with pytest.raises(ValueError, match="Camera gaps"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)


def test_noncritical_nonmonotonic_timestamps_are_still_rejected(setup_episode):
    dataset, root, cache, journal, sync, write_sync, stamps, _ = setup_episode
    sync["frame_records"][0]["host_frame_monotonic_ns"] = stamps[1]
    write_sync()
    with pytest.raises(ValueError, match="timestamps must increase"):
        import_dataset(dataset, root, cache, journal, critical_phase_only=True)
