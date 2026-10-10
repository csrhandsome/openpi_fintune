# Franka Panda 训练配置

训练环境使用 Python 3.12、PyTorch 和 `lerobot[dataset]==0.6.1`，读取采集端 v3.0 Parquet 数据。训练入口为 `scripts/train.py`，服务入口为 `scripts/server.py`；普通配置位于仓库根目录 `config/train/openpi.yaml`，RLT 配置位于 `config/train/openpi_rlt.yaml`。

## 字段与动作维度

Panda 数据集和服务请求使用点号字段：`observation.exterior_image`、`observation.wrist_image_left`、`observation.joint_position`、`observation.ee_pose`、`observation.gripper_position`；可选指内相机为 `observation.gripper_image_left/right`。

采集端分开保存 `action.joint_position` / `action.ee_pose` 与 `action.gripper_position`。YAML 的 `data.action_space` 选择 `joint`、`ee` 或 `ee_delta`；`data.action_dim` 是训练启动时检查的物理动作总维度。模型内部的 `model.action_dim` 是单独的 padding 维度。

| action_space | 数据集目标 | 训练表示 | 单臂默认 action_dim |
| --- | --- | --- | --- |
| joint | 绝对关节角 + 夹爪 | 绝对关节角 + 夹爪 | 8 |
| ee | 绝对 XYZ/RPY + 夹爪 | 绝对 XYZ/RPY + 夹爪 | 7 |
| ee_delta | 绝对 XYZ/RPY + 夹爪 | 相对当前 EE 的位姿增量 + 绝对夹爪 | 7 |

双臂 EE 设 `data.action_dim: 14`，字段为 `ee_pose[12]` 与 `gripper_position[2]`，转换后按每臂 7 维排列。joint 每臂关节维度可用 `data.motion_dim` 指定，默认 7；EE 为 6。启动首先读取 `meta/info.json` 校验 action 维度、各字段形状、fps，再加载模型和权重。

XYZ 用米，RPY 用弧度。EE-delta 训练折返姿态差，推理还原绝对位姿。夹爪为连续比例，0 关闭、1 张开。

每行目标对应下一帧新鲜相机观测，终止行使用最终实测状态。sidecar 应写 `action_semantics: next_camera_observation_measured_state` 或 `action_target_offset_frames: 1`，其他时序会被拒绝。

## 路径与训练

`data.repo_id` 使用完整 ID，包含采集日期后缀。默认数据根目录为相邻 `data_collect/data/dataset`，与工作目录无关。`HF_LEROBOT_HOME` 覆盖数据主目录；`OPENPI_DATASET_ROOT` 覆盖完整数据集路径。优先读取 `<home>/<repo ID basename>`，兼容旧 `<home>/<org>/<dataset>` 目录。

YAML 中的预训练权重和输出路径相对仓库根目录解析。先设置 `data.repo_id`。`pytorch_weight_path` 下没有 Torch 权重时，训练会从 `jax_weight_path` 的官方 JAX checkpoint 转一次。再用同一配置计算归一化和训练：

```bash
uv run scripts/compute_norm_stats.py --config config/train/openpi.yaml
uv run scripts/train.py --config config/train/openpi.yaml
uv run scripts/server.py --config config/train/openpi.yaml --port 8001
```

默认普通 VLA 输出为 `checkpoints/openpi/ee/<step>`，归一化为 `data/assets/openpi/<repo_id>`。每次运行和保存的 checkpoint 都有实际生效的 `config.yaml`。

已有 EE VLA 的 readout 训练使用 `openpi_rlt.yaml` 并配置 `rlt.vla_checkpoint`，无需重新训练 VLA。主服务可以同时输出动态 token 和 EE 参考；后续 B/C actor 的关节参考协议见 [Force RLT](force_rlt_pytorch.md)。
