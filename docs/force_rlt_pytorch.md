# Force RL Token：PyTorch 插入任务实现

本实现对应仓库根目录的 `FORCE_RLTOKEN_MIGRATION_PLAN.md`。离线 readout 训练使用 `pi0_rlt` 的 EE PyTorch checkpoint（30 Hz、16 步，6D 基座系位姿 + 夹爪，共 7D 动作）：VLA 只接收外部、腕部相机、EE 位姿和夹爪状态；左右指内 RGB 只进入接触 actor。腕部相机是当前 Franka checkpoint 的已有视角，A0/A1/B/C 保持一致。旧 JAX ForceVLA 路径已移除；这个实验使用 PyTorch Pi0 底座。

## 代码路径

| 路径 | 作用 |
| --- | --- |
| `models_pytorch/pi0_pytorch.py` | 提取 post-transformer image prefix，并复用同一前缀 KV cache 生成 VLA 动作。原 `sample_actions` 与新路径用同一噪声逐值对齐。 |
| `models_pytorch/rl_token.py` | 单独训练的交叉注意力压缩和遮罩重建；VLA 的 prefix 在训练时停止梯度。 |
| `scripts/train_rlt_pytorch.py` | 训练 readout；保存权重、配置和训练协议。 |
| `scripts/train_force_rlt_pytorch.py`、`scripts/serve_force_rlt_pytorch.py` | 从回放分别训练 B/C actor；单个 WebSocket 服务加载 VLA、readout 和 actor，一次请求返回 `z_rlt`、`ref_q`、`ref_chunk` 和 `proposed_q`。 |
| `force_rlt/tactile_encoder.py`、`observation.py` | 同权 ResNet-18、固定 RGB/ROI/ImageNet 预处理、左右分开编码、每回合无接触基线、四帧历史和有效位。 |
| `force_rlt/networks.py`、`learner.py`、`replay.py` | PyTorch H 步 actor、只看 E 步实际动作的 twin critic、target 网络、带终止掩码的 TD、回放及独立 checkpoint。B 使用恒零指内特征，并须独立训练。 |
| `force_rlt/action_adapter.py`、`runtime.py` | 基于当前实测 TCP 的参考重锚定、IK、关节/TCP/工作区投影、连续帧交接、接管隔离。 |

## 准备与运行

1. 选定**不含指内图像**的 `pi0_rlt` PyTorch checkpoint，并确认其中有 `model.safetensors`、`assets/` 和可用的归一化统计。用同一个 checkpoint 和相机协议采集所有组。对基座先验证能稳定接近插入件，不满足时不要进入接触 RL。
2. 训练 readout：

   ```bash
   uv run python scripts/train_rlt_pytorch.py --config pi0_rlt \
     --vla-checkpoint /path/to/pytorch-checkpoint --output /path/to/readout --steps 5000
   ```

   readout 训练读取仓库的 PyTorch 数据迭代器，并强制使用指定 VLA checkpoint 的 `assets/` 归一化统计，对有效 image patch 重建；`manifest.json` 保存 token 形状、相机协议、训练遮罩协议和 `action_space=ee`、`physical_action_dim=7`。VLA 保持原 checkpoint；仅另存 readout 权重。推理按当前观测生成 `z_rlt`，EE 参考保存在 `ref_chunk[H,7]`，不生成关节参考 `ref_q`。真实使用前需要检查重建损失、token 非塌缩、跨条件可观测性以及原 VLA 动作对齐。

3. 以下 B/C actor 链路仍使用 7D 关节目标。`pi0_rlt` 的 EE 输出需要先接入 EE→关节参考适配器；当前合并服务会拒绝直接加载 EE readout。以下命令仅适用于另行注册、具有匹配 checkpoint 的关节空间配置，不能把 EE 动作直接当成关节角。用每组**单独**的回放和训练预算训练 B/C：

   ```bash
   uv run python scripts/train_force_rlt_pytorch.py --replay /path/to/B.jsonl \
     --group B --output /path/to/B_actor.pt --steps 10000 \
     --max-reference-offset-rad <空载确认的值>
   uv run python scripts/train_force_rlt_pytorch.py --replay /path/to/C.jsonl \
     --group C --output /path/to/C_actor.pt --steps 10000 \
     --max-reference-offset-rad <同一值>
   ```

   B/C 从不同随机初始化开始；B 在训练与推理中把完整指内特征置零。`ReplayJournal` 读取的 JSONL 每条记录必须包含实际执行的 E 步目标、每步奖励、有效 mask、下一状态、来源和安全标记。终止或接管后的奖励不会 bootstrap 到之前的自主动作。

4. 部署合并服务（B/C 各启动一个实例，使用对应 checkpoint）：

   ```bash
   uv run python scripts/serve_force_rlt_pytorch.py --config <joint-config> --group B \
     --actor-checkpoint /path/to/B_actor.pt --vla-checkpoint /path/to/pytorch-checkpoint \
     --readout /path/to/readout --port 8002
   uv run python scripts/serve_force_rlt_pytorch.py --config <joint-config> --group C \
     --actor-checkpoint /path/to/C_actor.pt --vla-checkpoint /path/to/pytorch-checkpoint \
     --readout /path/to/readout --port 8003
   ```

   上述关节空间 B/C 客户端一次发送 VLA 的外部/腕部 RGB、`observation.joint_position[7]`、夹爪状态、`prompt`、`monotonic_ts` 和实测 `O_T_TCP[4,4]`。C 支持服务端编码四帧双指 RGB（下文），也保留预计算 `z_grip[D_g]` 与全真 `finger_valid_mask[4]` 的旧接口；B 在服务端恒置零指内输入。响应同时包含 `z_rlt`、物理空间 `ref_q`、`ref_chunk`、时间戳和 `proposed_q[H,7]`。提案仍须在工程机完成硬件安全投影后才可下发。data_collect 客户端已接入 B/C 请求与异步四帧历史，Force 真机执行仍需接入 TCP 安全投影器。

## 已验证范围与上线前检查

运行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q scripts/tests/rlt_pytorch_smoke_test.py src/openpi/models_pytorch/pi0_pytorch_test.py src/openpi/models_pytorch/rl_token_test.py src/openpi/force_rlt -m 'not manual'` 检查 readout 训练/保存/server 加载/WebSocket 请求、共享前缀数值对齐、遮罩损失、E 步 TD 与回放语义。smoke test 使用小型本地 VLA 夹具，不能替代真实 Franka checkpoint 的吞吐和显存验收。此处禁用自动 pytest 插件是因为当前环境的 ROS pytest 插件依赖缺少 `lark`。

当前交付范围是 PyTorch 训练与 server 部署。`ContactPolicy`、`FrankaSafetyProjector`、`ReplayJournal` 等是离线训练和后续系统集成可复用的模块；仓库不提供新的 Franka 真机插入入口。方案第 5 节的硬件小试、在线数据采集和 A0/A1/B/C 统计依赖具体设备及数据，未在本仓库验证。


Panda 采集、训练和服务使用点号字段（LeRobot 不接受斜线）：
`observation.exterior_image`、`observation.wrist_image_left`、`observation.joint_position` /
`observation.ee_pose`、`observation.gripper_position`、`observation.gripper_image_left/right`。
训练从 `action.joint_position` / `action.ee_pose` 和 `action.gripper_position` 读取未来块，
只在模型输入时拼成内部 `actions`；无需复制采集端的 YAML。普通 DROID 数据配置保持原有接口。

C 使用采集端时，为服务命令添加 `--finger-weights /path/to/resnet18.pth` 和
`--finger-roi X0 Y0 X1 Y1`（默认全图 224×224），启用 `panda-finger-rgb-history-v1`。
客户端只采集四帧 RGB 与时间戳；GPU 服务端将基线和未缓存帧放到一个 batch 编码，
按 FingerHistory 的帧/左右指/绝对特征与两类差分顺序构造 `z_grip`。
同会话的重叠帧按时间戳缓存。四帧必须同步、唯一且新鲜；掩码不能以假数据补齐。
权重、采集的裁剪/缩放和 ROI 必须与 actor 训练一致。会话开始时需要张开且无接触的基线。
未设置 `--finger-weights` 时仍接受原来的预计算 `z_grip` C 请求。
协议不要求 `actor_sha256`。机器人控制循环不执行神经编码；实际请求延迟仍需真实 GPU checkpoint 验收。
