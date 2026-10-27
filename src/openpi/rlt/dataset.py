"""Adapt existing data_collect LeRobot episodes and frontend sync annotations."""

from dataclasses import asdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.rlt.episode import build_episode_transitions
from openpi.rlt.episode import finite_array
from openpi.rlt.episode import window_starts
from openpi.rlt.features import FeatureCache
from openpi.rlt.replay import ReplayJournal
from openpi.rlt.rewards import TerminalReward


def confined_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset sidecar path escapes dataset root")
    return path


def read_annotation(root: Path, episode_index: int) -> dict:
    for folder in ("audio", ""):
        path = confined_path(root, f"{folder}/episode_{episode_index:06d}.sync.json".lstrip("/"))
        if path.is_file():
            sync = json.loads(path.read_text())
            if sync.get("episode_index") != episode_index:
                raise ValueError("Sync sidecar episode mismatch")
            success = sync.get("success")
            if success is not None and type(success) is not bool:
                raise ValueError("success must be true, false or null")
            return sync
    return {"success": None}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _proprio(item: dict, space: str) -> np.ndarray:
    key = "observation.joint_position" if space == "joint" else "observation.ee_pose"
    return np.concatenate(
        (_numpy(item[key]).reshape(-1), _numpy(item["observation.gripper_position"]).reshape(-1))
    ).astype(np.float32)


def _quaternion_rpy(value) -> np.ndarray:
    q = finite_array(value, (4,), "target EE quaternion")
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("Target quaternion cannot be zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ],
        dtype=np.float32,
    )


def _source(row: dict, sync: dict) -> str:
    # These flags describe the actual controller, even when the proposal was RL.
    human = False
    for key in ("intervention_flag", "human_controlled"):
        if key in row:
            if type(row[key]) is not bool:
                raise ValueError(f"{key} must be an explicit boolean")
            human |= row[key]
    if human:
        return "HUMAN"
    name = row.get("control_source", row.get("source"))
    if name in {"BASE", "RL", "HUMAN"}:
        return name
    if name in {"human", "vr"}:
        return "HUMAN"
    if name == "actor":
        return "RL"
    if name == "policy":
        return "BASE"
    if name is not None:
        raise ValueError(f"Unknown per-step action source: {name}; use BASE, RL or HUMAN")
    if "vr_arm_enabled" in row:
        if type(row["vr_arm_enabled"]) is not bool:
            raise ValueError("vr_arm_enabled must be an explicit boolean")
        return "HUMAN" if row["vr_arm_enabled"] else "BASE"
    return "HUMAN" if sync.get("source", "collection") == "collection" else "BASE"


def _trace_rows(root: Path, sync: dict) -> list[dict]:
    if not sync.get("action_trace"):
        return []
    path = confined_path(root, sync["action_trace"])
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _numeric_items(dataset, start: int, stop: int, *, space: str, action_source: str) -> list[dict]:
    if not hasattr(dataset, "select_columns"):
        return [dataset[index] for index in range(start, stop)]
    target = "joint_position" if space == "joint" else "ee_pose"
    columns = ["timestamp", f"observation.{target}", "observation.gripper_position"]
    if action_source == "measured":
        columns.extend((f"action.{target}", "action.gripper_position"))
    numeric = dataset.select_columns(columns)
    return [numeric[index] for index in range(start, stop)]


def episode_actions(
    root: Path,
    sync: dict,
    items: list[dict],
    frame_ns: np.ndarray,
    *,
    space: str,
    action_source: str,
    max_skew_s: float = 0.05,
) -> tuple[np.ndarray, list[str]]:
    rows = _trace_rows(root, sync)
    indices = None
    if rows:
        times = np.asarray([int(row["host_sample_monotonic_ns"]) for row in rows], np.int64)
        if np.any(np.diff(times) <= 0):
            raise ValueError("Action trace timestamps must increase")
        # Last issued command strictly before each next observation; never use a future command.
        indices = np.searchsorted(times, frame_ns[1:], side="left") - 1
        if np.any(indices < 0):
            raise ValueError("Execution trace does not cover the observation intervals")
        skew = frame_ns[1:] - times[indices]
        if np.any(skew > max_skew_s * 1e9) or np.any(times[indices] < frame_ns[:-1]):
            raise ValueError("Execution trace has stale or missing commands")
    actions, sources = [], []
    for index, item in enumerate(items[:-1]):
        row = rows[int(indices[index])] if indices is not None else {}
        if action_source == "command":
            if not row:
                raise ValueError(
                    "Command replay requires an action trace; historical data may use explicit --action-source measured"
                )
            if space == "ee":
                pose = np.concatenate(
                    (
                        finite_array(row["target_ee_position"], (3,), "target EE position"),
                        _quaternion_rpy(row["target_ee_orientation_xyzw"]),
                    )
                )
            else:
                if "target_joint_position" not in row:
                    raise ValueError(
                        "Joint command replay requires recorded target_joint_position; measured states are not commands"
                    )
                pose = finite_array(row["target_joint_position"], (7,), "target joints")
            grip = row.get("target_gripper_position", row.get("gripper_position"))
            if grip is None:
                raise ValueError("Execution trace lacks the commanded gripper ratio")
            action = np.concatenate((pose, np.asarray(grip, np.float32).reshape(-1)))
        elif action_source == "measured":
            key = "action.joint_position" if space == "joint" else "action.ee_pose"
            action = np.concatenate(
                (_numpy(item[key]).reshape(-1), _numpy(item["action.gripper_position"]).reshape(-1))
            )
        else:
            raise ValueError("action_source must be command or measured")
        actions.append(action)
        sources.append(_source(row, sync))
    return np.asarray(actions, np.float32), sources


def import_dataset(
    dataset,
    root: Path,
    cache: FeatureCache,
    journal: ReplayJournal,
    *,
    reward: TerminalReward | None = None,
    stride: int = 2,
    action_source: str = "command",
    episode_indices: list[int] | None = None,
    critical_phase_only: bool = False,
) -> dict:
    """Replace annotated episodes; optional critical scope excludes every interval crossing the marker.

    Critical scope requires versioned absolute-clock phase annotations. Missing/null markers remove
    any prior entry instead of promoting legacy or unmarked episodes into the contact replay.
    """
    root = Path(root).resolve()
    reward = reward or TerminalReward()
    spec = journal.spec
    phase_scope = "critical_phase" if critical_phase_only else "full_episode"
    provenance = {"model_id": cache.model_id, "action_source": action_source}
    if critical_phase_only:
        provenance["phase_scope"] = phase_scope
    journal.bind_provenance(provenance)
    if not np.isclose(float(dataset.meta.fps), spec.fps):
        raise ValueError("Dataset and replay action timebases differ; align the dataset before importing")
    identity = root.name + ":" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    stats = {
        "imported_episodes": 0,
        "skipped_unannotated": 0,
        "skipped_no_critical_phase": 0,
        "skipped_empty_critical_phase": 0,
        "excluded_noncritical_steps": 0,
        "included_steps": 0,
        "transitions": 0,
    }
    for episode in dataset.meta.episodes:
        index = int(episode["episode_index"])
        if episode_indices is not None and index not in episode_indices:
            continue
        episode_id = f"{identity}:{index}"
        sync = read_annotation(root, index)
        if sync.get("success") is None:
            journal.remove_episode(episode_id)
            stats["skipped_unannotated"] += 1
            continue
        start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
        marker = sync.get("critical_phase_start_monotonic_ns")
        if critical_phase_only:
            if marker is None:
                journal.remove_episode(episode_id)
                stats["skipped_no_critical_phase"] += 1
                stats["excluded_noncritical_steps"] += max(stop - start - 1, 0)
                continue
            if (
                type(sync.get("critical_phase_annotation_version")) is not int
                or sync["critical_phase_annotation_version"] != 1
            ):
                raise ValueError("Critical-phase replay requires critical_phase_annotation_version 1")
            if type(marker) is not int or marker < 0:
                raise ValueError("critical_phase_start_monotonic_ns must be a nonnegative integer or null")
        items = _numeric_items(dataset, start, stop, space=spec.action_space, action_source=action_source)
        if len(items) < 2:
            raise ValueError(f"Episode {index} has no observed next-state transition")
        frames = sync.get("frame_records", [])
        if frames and len(frames) != len(items):
            raise ValueError("Sync frame count disagrees with LeRobot episode")
        frame_ns = np.asarray(
            [int(frame["host_frame_monotonic_ns"]) for frame in frames]
            if frames
            else [round(float(_numpy(item["timestamp"]).item()) * 1e9) for item in items],
            np.int64,
        )
        if np.any(np.diff(frame_ns) <= 0):
            raise ValueError("Camera frame timestamps must increase")
        if sync.get("action_trace") and not frames:
            raise ValueError("Trace alignment requires absolute sync frame timestamps")
        phase_start = 0
        if critical_phase_only:
            if not frames:
                raise ValueError("Critical-phase alignment requires absolute sync frame timestamps")
            # A command interval that starts before the marker is excluded in full.
            # The suffix still ends at the original last state, preserving terminal semantics.
            phase_start = int(np.searchsorted(frame_ns, marker, side="left"))
            stats["excluded_noncritical_steps"] += min(phase_start, len(items) - 1)
            if phase_start >= len(items) - 1:
                journal.remove_episode(episode_id)
                stats["skipped_empty_critical_phase"] += 1
                continue
        observed_frames = len(items)
        items = items[phase_start:]
        frame_ns = frame_ns[phase_start:]
        intervals = np.diff(frame_ns) / 1e9
        if np.any(np.abs(intervals - 1 / spec.fps) > 0.5 / spec.fps):
            raise ValueError("Camera gaps would change chunk discount timing; align or split the episode first")
        actions, sources = episode_actions(
            root, sync, items, frame_ns, space=spec.action_space, action_source=action_source
        )
        finite_array(actions, (len(items) - 1, spec.action_dim), "episode actions")
        starts = window_starts(len(actions), chunk_len=spec.chunk_len, stride=stride)
        required = set(starts) | {min(start + spec.chunk_len, len(actions)) for start in starts}
        features = [None] * len(items)
        for frame_index in sorted(required):
            item = dataset[start + phase_start + frame_index]
            observation = {key: _numpy(value) for key, value in item.items() if key.startswith("observation.")}
            observation.update(
                prompt=str(item.get("task", sync.get("task", ""))), monotonic_ts=np.asarray(frame_ns[frame_index] / 1e9)
            )
            proprio = finite_array(_proprio(item, spec.action_space), (spec.proprio_dim,), "proprio")
            features[frame_index] = cache.get(observation, proprio)
        records = build_episode_transitions(
            episode_id, features, actions, sources, spec=spec, success=sync["success"], stride=stride, reward=reward
        )
        records = [replace(record, step_id=record.step_id + phase_start) for record in records]
        journal.replace_episode(
            records,
            metadata={
                "dataset_root": str(root),
                "episode_index": index,
                "saved_at_ns": sync.get("saved_at_ns"),
                "success": sync["success"],
                "reward": asdict(reward),
                "stride": stride,
                "action_source": action_source,
                "model_id": cache.model_id,
                "reference_origin": "offline_vla_recomputed",
                "phase_scope": phase_scope,
                "critical_phase_annotation_version": sync.get("critical_phase_annotation_version"),
                "critical_phase_start_monotonic_ns": marker,
                "first_included_step": phase_start,
                "excluded_noncritical_steps": phase_start,
                "observed_frames": observed_frames,
                "included_frames": len(items),
                "original_executed_steps": observed_frames - 1,
                "executed_steps": len(actions),
            },
        )
        stats["imported_episodes"] += 1
        stats["included_steps"] += len(actions)
        stats["transitions"] += len(records)
    return stats
