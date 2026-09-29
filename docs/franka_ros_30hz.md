# Franka ROS 30 Hz training

The ROS collector stores both `ee_pose` and `joint_position` observations in
every LeRobot frame. Its `dataset.action_space` YAML setting chooses `actions`:

| `action_space` | LeRobot `actions` | Training config |
| --- | --- | --- |
| `ee` | `[x, y, z, roll, pitch, yaw, gripper_open]` (7D) | `pi0_base_ee_ros_30hz` |
| `joint` | `[joint1, ..., joint7, gripper_open]` (8D) | `pi0_base_joint_ros_30hz` |

XYZ is in metres in the robot base frame. Roll, pitch and yaw are ZYX Euler
angles in radians. Gripper is a logical 0/1 open state. Each frame's `actions`
is the measured arm and gripper state at the next fresh camera pair, or the
final measured state at the end of an episode. The first action in a training
chunk therefore targets one 30 Hz step after the observation. The LeRobot
loader reads `actions` directly with offsets 0 through 15. The 100 Hz
`episode_XXXXXX.actions.jsonl` trace is retained for diagnostics and is not
used by these training configs.

Run from this repository in the GPU environment with the collector's `data/`
directory and the complete LeRobot repo id, including its date suffix:

```bash
bash scripts/train_franka_ros_30hz.sh \
  /data/codes/grade_two/my_projects/franka_control_ros2/data \
  openpi/franka_lerobot_test \
  ros_30hz
```

The script checks the 30 Hz data and all episode timing metadata, chooses the
EE or joint configuration from the action dimension, computes normalization
statistics, and trains. Use a new dataset repo id/date after changing the
action space or the frame action timing. Old recordings without
`action_target_offset_frames: 1` cannot use this direct-action training path.

Serve an EE checkpoint for the ROS `inference.py` client:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_base_ee_ros_30hz \
  --policy.dir=checkpoints/pi0_base_ee_ros_30hz/ros_30hz/<step> \
  --port=8001
```

The plain ROS inference client expects EE metadata and 7D actions. A joint
checkpoint needs a separate joint action executor.
