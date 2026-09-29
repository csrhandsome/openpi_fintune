#!/usr/bin/env bash
# Usage: scripts/train_franka_ros_30hz.sh <collector-data-dir> <repo-id> [experiment-name]
# Example: scripts/train_franka_ros_30hz.sh \
#   /data/codes/grade_two/my_projects/franka_control_ros2/data openpi/franka_lerobot_test ros_30hz
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: $0 <collector-data-dir> <repo-id> [experiment-name]" >&2
    exit 2
fi

project_root="$(cd "$(dirname "$0")/.." && pwd)"
export HF_LEROBOT_HOME="$(realpath "$1")"
export FRANKA_ROS_DATASET_REPO_ID="$2"
dataset_dir="$HF_LEROBOT_HOME/$FRANKA_ROS_DATASET_REPO_ID"
experiment_name="${3:-ros_30hz}"

if [[ ! -f "$dataset_dir/meta/info.json" ]]; then
    echo "LeRobot dataset not found: $dataset_dir" >&2
    exit 1
fi
config_name="$(python3 - "$dataset_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
features = info["features"]
if abs(float(info["fps"]) - 30.0) > 0.01:
    raise SystemExit("Franka ROS training requires a 30 Hz LeRobot dataset")
action_shape = features["actions"]["shape"]
if action_shape == [7] and features.get("ee_pose", {}).get("shape") == [6]:
    action_space = "ee"
    config_name = "pi0_base_ee_ros_30hz"
elif action_shape == [8] and features.get("joint_position", {}).get("shape") == [7]:
    action_space = "joint"
    config_name = "pi0_base_joint_ros_30hz"
else:
    raise SystemExit(f"Unsupported Franka observation/action features: {action_shape}")
episodes = int(info["total_episodes"])
if episodes < 1:
    raise SystemExit("Dataset has no saved episodes")
for episode in range(episodes):
    path = root / f"episode_{episode:06d}.sync.json"
    if not path.is_file():
        raise SystemExit(f"Missing action timing metadata: {path}")
    sync = json.loads(path.read_text(encoding="utf-8"))
    if sync.get("action_space") != action_space or sync.get("action_target_offset_frames") != 1:
        raise SystemExit(f"Action timing or space mismatch in {path}; record a new dataset")
print(config_name)
PY
)"

cd "$project_root"
echo "Training $config_name from LeRobot actions in $dataset_dir"
uv run scripts/compute_norm_stats.py --config-name "$config_name"
uv run scripts/train.py "$config_name" --exp_name "$experiment_name" --num_workers 4
