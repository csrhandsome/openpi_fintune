# Franka Panda 30 Hz training

The training/server environment uses Python 3.12 and `lerobot[dataset]==0.6.1`,
matching the collector's v3.0 Parquet dataset format. Install with `uv sync --locked`.
Both training and the normal policy server default to `pi0_base_ee_ros_30hz`.

The collector uses the observation/action fields in `data_collect/config/dataset/panda.yaml`.
LeRobot disallows slashes in feature names, so dataset columns and Panda server requests
use `observation.exterior_image`, `observation.wrist_image_left`, `observation.joint_position`,
`observation.ee_pose`, `observation.ee_position`, `observation.gripper_position` and optional
`observation.gripper_image_left` / `observation.gripper_image_right`.

The collector stores `action.joint_position[7]`, `action.ee_pose[6]` and
`action.gripper_position[1]` separately. `pi0_rlt` and `pi0_base_ee_ros_30hz`
read pose/gripper targets. The input transforms concatenate those targets
into the model's internal 7D `actions`. No dataset YAML is loaded in this repository.
XYZ uses metres and ZYX roll/pitch/yaw uses radians; gripper ratio is continuous, 0 closed and 1 open.
Each row targets the next fresh camera observation; the terminal row uses the final measured state.
The sync sidecar declares `action_semantics: next_camera_observation_measured_state`.
An explicit `action_target_offset_frames: 1` is also accepted; other offsets are rejected.

Create a new dataset date/root for the new columns. Old flat-column datasets require conversion.
The 100 Hz action trace remains available for optional multirate alignment.

Use the complete repo ID, including the collector's date suffix. Like the collector,
the loader uses `<dataset home>/<repo ID basename>` without an organization directory.
The default home is the sibling `data_collect/data/dataset` directory, resolved from
the source location rather than the current working directory. `HF_LEROBOT_HOME`
overrides that home; `OPENPI_DATASET_ROOT` overrides the complete dataset path.
Existing nested `<home>/<org>/<dataset>` directories remain a fallback when no flat
dataset exists. Task prompts are read from
`meta/tasks.parquet`, and episode boundaries from the v3 episode metadata.

Set `OPENPI_PYTORCH_WEIGHT_PATH` to an RLinf-layout Pi0 checkpoint. Run normalization
and training with the same EE config:

```bash
# Defaults already select ../data_collect/data/dataset/franka_lerobot_test_100hz.
# Set HF_LEROBOT_HOME and FRANKA_ROS_DATASET_REPO_ID for another dataset.
export OPENPI_PYTORCH_WEIGHT_PATH=/path/to/model.safetensors
uv run scripts/compute_norm_stats.py --config-name pi0_base_ee_ros_30hz
uv run scripts/train.py --exp-name ros_30hz --num-workers 4
```

For offline RL-token readout training, use `pi0_rlt` with an existing EE checkpoint
as documented in `force_rlt_pytorch.md`. To train its VLA first, select the same
`pi0_rlt` config for both normalization and `scripts/train.py`.
The normal configs read the next-frame labels at 30 Hz.

Serve the latest EE checkpoint with `uv run scripts/serve_policy.py --port 8001`.
Set `OPENPI_SERVER_CONFIG` and `OPENPI_SERVER_CHECKPOINT_DIR` to select another config
and checkpoint. The EE server returns `actions[16,7]`: absolute base-frame pose
and gripper ratio, with metadata `action_space=ee`, `action_fps=30`, `action_horizon=16`.
The client must use the same action space, fps and horizon. The B/C actor server
has its own joint-target protocol; see `force_rlt_pytorch.md`.
