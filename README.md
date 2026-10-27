# openpi-force

本仓库基于 Physical Intelligence 的 openpi，使用 RLinf 对齐的 PyTorch Pi0/Pi05 训练 VLA，并提供 readout 与可选的 VLA/readout 联合训练。任务参数放在仓库根目录 `config/train/`；RL-token 第一阶段默认 `rlt.alpha: 0`，冻结 VLA，只训练 readout，详见 [实现对齐说明](docs/pytorch_alignment.md)。

## 环境

Python 3.12，LeRobot 0.6.1（v3.0 Parquet 数据集），PyTorch 2.7.1。

```bash
uv sync --locked
```

数据读取统一使用 LeRobot v3。SwanLab 0.10.3 或更新版本随主依赖安装，默认启用 `dev` 依赖组。

底层共享模型和变换仍有 JAX 兼容依赖；训练、优化器、数据加载和 checkpoint 保存使用 PyTorch。

在训练 YAML 的 `train.tracking_backend` 中选择日志平台：

```yaml
train:
  tracking_backend: swanlab  # swanlab / wandb / none（关闭追踪）
```

两份训练配置默认选择 `swanlab`。也可用 `--tracking-backend wandb` 或 `--tracking-backend none` 临时覆盖。首次使用对应平台前执行 `uv run swanlab login` 或 `uv run wandb login`；也支持 `SWANLAB_API_KEY` / `WANDB_API_KEY` 环境变量，密钥无需写入 YAML。

`project_name` 指定项目，`exp_name` 指定实验名。普通 VLA 和 RLT readout（`stage: sft`）记录训练指标；VLA 还记录相机样本和 checkpoint 步数。DDP 仅主进程上传。续训读取输出目录中对应的 `swanlab_id.txt` 或 `wandb_id.txt`；切换平台且没有该平台的 ID 时新建实验，保留另一平台的 ID。

日志统一放在 `data/`：`data/swanlab/<name>/<exp_name>/`、`data/wandb/<name>/<exp_name>/`，SwanLab SDK 默认日志为 `data/swanlog/`，均由 `data/` 忽略规则覆盖。旧 checkpoint YAML 中的 `wandb_enabled` / `swanlab_enabled` 自动转换为相应平台或关闭状态，新保存的配置统一使用 `tracking_backend`。

## 配置与入口

| 配置 | 用途 | 默认保存目录 |
| --- | --- | --- |
| `config/train/openpi.yaml` | 普通 VLA 训练，`mode: vla` | `checkpoints/openpi/ee/<step>/` |
| `config/train/openpi_rlt.yaml` | `mode: rlt`；`stage: sft / actor_critic / force` 选择 readout、普通 RL 或接触 actor | `checkpoints/pi0_rlt/readout/` 或 `runs/openpi_rlt/` |
| `config/server/openpi.yaml` | 普通 VLA 推理，引用训练 YAML | — |
| `config/server/openpi_rlt.yaml` | RLT 在线训练/actor 评估、可选触觉输入 | `runs/openpi_rlt/` |

YAML 记录模型结构、数据集、动作表示、物理动作维度、预训练权重、batch、学习率、训练步数、日志/保存间隔、精度和输出路径。相对路径均从仓库根目录解析。需要另一个实验时，在 `config/train/` 复制一份 YAML 并修改参数。

`scripts/train.py` 读取训练配置后选择 VLA、readout、普通 actor–critic 或 force 接触训练；`scripts/server.py` 读取 `config/server/` 里的服务 YAML，引用训练配置组装普通 VLA、RLT 或 force 服务。显式传入 `stage: force` 的训练 YAML 也会直接组装接触服务。具体实现位于 `src/openpi/training/` 和 `src/openpi/serving/`。

## 动作表示与维度

在 YAML 的 `data` 下设置：

```yaml
data:
  type: franka
  action_space: ee_delta  # joint / ee / ee_delta
  action_dim: 7          # 物理动作维度，训练开始时检查
  motion_dim: null       # 每臂默认：EE 为 6，joint 为 7
  gripper_dim: 1         # 每臂夹爪维度
```

`joint` 使用绝对关节位置目标；`ee` 使用绝对 XYZ/RPY 目标；`ee_delta` 从数据集的绝对 XYZ/RPY 转换为相对当前观测的增量，RPY 差值折返到 `[-π, π)`。EE-delta 推理还原为绝对位姿，夹爪始终保持绝对值。

EE 单臂为 `6+1=7`，双臂为 `(6+1)*2=14`。双臂数据字段分别为 `action.ee_pose[12]` 和 `action.gripper_position[2]`，输入时组合成每臂连续的 `[pose, gripper]` 块。joint 单臂默认 `7+1=8`。启动时先校验数据集 `meta/info.json` 中的真实 action 维度；不匹配时在加载模型权重之前报错。

`model.action_dim: 32` 表示模型内部 padding，必须不小于 `data.action_dim`，不作为数据集动作维度。

## 普通 VLA 训练

先在 `openpi.yaml` 设置 `data.repo_id`。`pytorch_weight_path` 默认是 `data/checkpoints/torch/pi0_base`。那里没有 RLinf 布局的 `model.safetensors` 时，训练会用 RLinf 的转换脚本把 `jax_weight_path`（默认官方 `gs://openpi-assets/checkpoints/pi0_base`）转一次再开始微调；已有文件则直接加载。数据集路径由 `HF_LEROBOT_HOME` 或 `OPENPI_DATASET_ROOT` 控制，详见 [Franka 数据说明](docs/franka_ros_30hz.md)。

```bash
uv run scripts/compute_norm_stats.py --config config/train/openpi.yaml
uv run scripts/train.py --config config/train/openpi.yaml
```

支持临时覆盖运行参数，实际生效的配置会保存到运行目录和每个 checkpoint 的 `config.yaml`：

```bash
uv run scripts/train.py --config config/train/openpi.yaml --batch-size 8 --exp-name example
uv run scripts/train.py --config config/train/openpi.yaml --exp-name example --resume
uv run scripts/train.py --config config/train/openpi.yaml --num-gpus 2 --batch-size 8
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config config/train/openpi.yaml --num-gpus 2 --batch-size 8
```

`model.is_lora: true` 时，用 `model.lora_mode` 选择训练范围：

| 设置 | Gemma 语言模型与动作专家 | SigLIP | 动作、状态、时间投影 |
| --- | --- | --- | --- |
| `lora_mode: lora_only`（默认） | PEFT LoRA，基础权重冻结 | 线性层 PEFT LoRA，基础权重冻结 | PEFT LoRA，基础权重冻结 |
| `lora_mode: openpi` | PEFT LoRA，基础权重冻结 | 更新原始权重 | 更新原始权重 |

例如 `uv run scripts/train.py --config config/train/openpi.yaml --model.lora-mode openpi`。`openpi` 模式兼容官方 JAX 示例的视觉/投影训练范围，Gemma adapter 仍使用当前 PEFT 实现，不是官方 JAX 原生 LoRA 的逐算子移植。`model.lora_rank` 控制两种模式的 PEFT rank；`model.is_lora: false` 时全参训练。完整 PEFT checkpoint 必须使用原来的模式和 rank 加载、续训；切换模式应从 dense 基础权重开启新实验。

`train.batch_size` 是每次优化器更新的全局 batch。普通 VLA 的 `train.micro_batch_size` 默认 `null`：每卡一次处理 `batch_size / num_gpus` 个样本，反传一次就更新；单卡默认一次处理全部 5 条。显式设为整数才指定每卡一次前向的 batch，并按 `batch_size / (micro_batch_size * num_gpus)` 次累积后更新，要求整除。梯度累积可用于单卡或多卡：batch 5、micro batch 1、单卡时累积五次；batch 8、micro batch 1、两卡时累积四次。`train.num_gpus` 控制自动启动的 DDP 进程数；每卡保留完整模型，DDP 不切分模型权重。普通 checkpoint 包含 `model.safetensors`、`optimizer.pt`、`metadata.pt`、`config.yaml` 和归一化资产。

## 离线 RL-token 训练与服务

在 `openpi_rlt.yaml` 设置 `rlt.vla_checkpoint`，并保持模型结构和动作表示与该 VLA 原训练一致。readout 强制读取该 checkpoint 的 `assets/` 归一化统计，只更新 readout 权重。

```bash
uv run scripts/train.py --config config/train/openpi_rlt.yaml
uv run scripts/server.py --config config/train/openpi_rlt.yaml --port 8001
```

readout 输出目录包含配套 EMA `model.safetensors`、`readout.safetensors`、`manifest.json` 和 `config.yaml`；定期与结束时保存步骤子目录及可恢复的 `training_state.pt`。推理按当前观测生成 `z_rlt`，并返回物理空间的 `ref_chunk`。readout 第一阶段支持 DDP 和 `train.resume: true`；`rlt.alpha > 0` 时联合训练 VLA。

普通 VLA 服务默认选取 YAML 保存目录中的最新 PyTorch checkpoint；也可指定现有 checkpoint：

```bash
uv run scripts/server.py --config config/server/openpi.yaml --checkpoint /path/to/checkpoint --port 8001
```

后续接触 actor 的代码和协议见 [Force RLT](docs/force_rlt_pytorch.md)。旧的上游环境配置名称和 JAX checkpoint 不能直接传给新的训练/服务入口。

RLT 在线模式默认由 VLA 收集经验，至少 600 条已标注 transition、20000 次 learner 更新且 actor 快照就绪后，在下一 episode 切换到 actor。`rlt.mode: eval` 固定 actor 权重；train/inference 共用 `rlt.force.enabled`。启动、执行反馈协议与客户端示例见 [RLT 在线说明](docs/rlt_online.md)。

## 测试

全部测试集中在 `src/openpi/tests/`，按 training、serving、policies、models、models_pytorch、rlt、shared、client 分目录。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest src/openpi/tests/training src/openpi/tests/serving \
  src/openpi/tests/models_pytorch src/openpi/tests/rlt/force_rlt -m 'not manual'
```

禁用自动插件可避免本机 ROS pytest 插件的额外依赖。真实设备连接检查为手动工具：`python -m openpi.tests.manual.connection`。

普通 RLT 的 episode 奖励、Pickle replay 构建与独立 actor/critic 训练见 [RLT 使用说明](src/openpi/rlt/README.md)。
