# Force RL Token：PyTorch 插入任务实现

本实现对应仓库根目录的 `FORCE_RLTOKEN_MIGRATION_PLAN.md`。离线 readout 训练使用 `pi0_rlt` 的 EE PyTorch checkpoint（30 Hz、16 步，6D 基座系位姿 + 夹爪，共 7D 动作）：VLA 只接收外部、腕部相机、EE 位姿和夹爪状态；左右指内 RGB 只进入接触 actor。腕部相机是当前 Franka checkpoint 的已有视角，A0/A1/B/C 保持一致。旧 JAX ForceVLA 路径已移除；这个实验使用 PyTorch Pi0 底座。

## 代码路径

| 路径 | 作用 |
| --- | --- |
| `models_pytorch/pi0_pytorch.py` | 提取 post-transformer image prefix，并复用同一前缀 KV cache 生成 VLA 动作。原 `sample_actions` 与新路径用同一噪声逐值对齐。 |
| `models_pytorch/rl_token.py` | 交叉注意力压缩和重建；readout 重建损失不向 VLA prefix 反传，默认 sft 通过 VLA 动作损失联合训练。 |
| `scripts/train.py` + `training/rlt.py` | 训练 readout；保存权重、配置和训练协议。 |
| `rlt/force_rlt/train.py`、`rlt/force_rlt/serve.py` | 通过 `stage: force` 和指内开关训练独立接触 actor；关节空间专用 WebSocket 服务加载 VLA、readout 和 actor，一次请求返回 `z_rlt`、`ref_q`、`ref_chunk` 和 `proposed_q`。 |
| `rlt/force_rlt/tactile_encoder.py`、`observation.py` | 同权 ResNet-18、固定 RGB/ROI/ImageNet 预处理、左右分开编码、每回合无接触基线、四帧历史和有效位。 |
| `rlt/force_rlt/networks.py`、`learner.py`、`replay.py` | PyTorch H 步 actor、只看 E 步实际动作的 twin critic、target 网络、带终止掩码的 TD、回放及独立 checkpoint。`force.enabled: false` 使用恒零指内特征，并须独立训练。 |
| `rlt/force_rlt/action_adapter.py`、`runtime.py` | 基于当前实测 TCP 的参考重锚定、IK、关节/TCP/工作区投影、连续帧交接、接管隔离。 |

## 准备与运行

1. 选定**不含指内图像**的 `pi0_rlt` PyTorch checkpoint，并确认其中有 `model.safetensors`、`assets/` 和可用的归一化统计。用同一个 checkpoint 和相机协议采集所有组。对基座先验证能稳定接近插入件，不满足时不要进入接触 RL。
2. 配置根目录 `config/train/openpi_rlt.yaml`（模型结构、动作表示和动作维度须匹配已有 VLA），通过统一入口训练 readout：

   ```bash
   uv run scripts/train.py --config config/train/openpi_rlt.yaml \
     --rlt.vla-checkpoint /path/to/pytorch-checkpoint --rlt.output-dir /path/to/readout --num-train-steps 5000
   ```

   默认 sft 联合训练 VLA/readout，强制使用指定 VLA checkpoint 的 `assets/` 归一化统计，按代码默认的无 mask 协议重建 image prefix；保存匹配的 EMA VLA/readout 和可恢复训练状态。`manifest.json` 记录 token 形状、相机协议和动作布局。当前默认 EE 配置的参考为 `ref_chunk[H,7]`，不生成关节参考 `ref_q`；接触 actor 必须另用匹配的关节配置训练 readout。真实使用前需要检查重建损失、token 非塌缩、跨条件可观测性以及配套 VLA 的动作表现。

   动态 token + EE 参考服务使用 `uv run scripts/server.py --config config/train/openpi_rlt.yaml`，权重路径和 readout 目录由 YAML 指定。

3. 接触 actor 现在使用统一 YAML 入口。设置 `rlt.stage: force`，并把 `data.action_space` 改为 `joint`、`data.action_dim` 改为 `8`（七关节 + 夹爪）。必须使用匹配的关节 VLA/readout；当前默认 EE 配置不能直接使用，入口会拒绝。`force` 是独立的训练路径，开启指内特征不会自动切换 stage，也不会串行执行两组对照。

   ```yaml
   data:
     type: franka
     action_space: joint
     action_dim: 8
     # repo_id、assets 等保持与关节 VLA 一致
   rlt:
     stage: force
     vla_checkpoint: /path/to/joint-vla
     force:
       enabled: true
       input: features    # rgb 时服务端编码，须设置 weights、ROI 和时序限制
       feature_dim: 12288
     force_stage:
       policy: actor
       replay_path: /path/to/contact-replay.jsonl
       checkpoint_path: /path/to/force-actor.pt
       readout_dir: /path/to/joint-readout
       batch_size: 64
       max_reference_offset_rad: 0.05 # 按设备确认的幅度设置
   ```

   完整可调参数见 `config/train/openpi_rlt.yaml`。使用直观开关选择对照：

   | 配置 | 行为 |
   | --- | --- |
   | `force_stage.policy: actor`，`force.enabled: true` | 独立训练小 actor，使用真实指内特征 |
   | `force_stage.policy: actor`，`force.enabled: false` | 独立训练小 actor，训练和服务都把完整指内特征置零；网络仍保留相同特征维度 |
   | `force_stage.policy: vla` | 服务直接返回冻结 VLA 的关节参考，不加载 actor、不读取指内输入；无需执行训练命令 |

   ```bash
   uv run scripts/train.py --config config/train/openpi_rlt.yaml
   ```

   每个对照分别设置 replay 和 checkpoint 路径、使用独立初始化。总更新次数、种子和日志/保存间隔使用顶层 `num_train_steps`、`seed`、`log_interval`、`save_interval`；网络、batch 和优化参数使用 `rlt.force_stage`，独立于普通 `rlt.actor/critic/learner`。`rlt.actor_checkpoint` 可指定续训权重，`resume: true` 默认恢复 `force_stage.checkpoint_path`。续训校验网络、学习参数和指内开关，禁止把恒零 checkpoint 当成真实特征 checkpoint。

   `ReplayJournal` 读取的 JSONL 每条记录必须包含实际执行的 E 步关节目标、逐步奖励、有效 mask、下一状态、来源和安全标记。H 使用 YAML 的模型 horizon，E 从回放推导；token 和指内维度必须匹配 YAML。人工接管数据不进入自主 RL。`rlt.replay.directory` 的普通 Pickle replay 不能用于这条链路。

4. 使用同一份训练 YAML 部署，服务自动识别 `stage: force`，无需额外传 `--group`：

   ```bash
   uv run scripts/server.py --config config/train/openpi_rlt.yaml --port 8002
   ```

   也可以把 `config/server/openpi_rlt.yaml` 的顶层 `mode` 设置为 `force`，通过 `train_config` 引用上述训练 YAML。`checkpoint`、`readout` 和服务端口等仍可在 inference YAML 中覆盖；指内输入和 actor/vla 选择统一来自训练 YAML 的 `rlt.force` 和 `rlt.force_stage`。服务检查 actor 的指内开关、维度、horizon 与 readout 的关节动作协议。旧 `python -m openpi.rlt.force_rlt.train/serve --group ...` 入口保留兼容，但新 YAML 配置、元数据和响应使用 `force_enabled`、`force_policy` 表示选择。

   actor 请求一次发送 VLA 外部/腕部 RGB、`observation.joint_position[7]`、夹爪状态、`prompt`、`monotonic_ts` 和实测 `O_T_TCP[4,4]`。真实指内输入支持服务端编码四帧双指 RGB，也支持 `z_grip[D_g]` 与全真 `finger_valid_mask[4]`；恒零对照无需指内输入。响应包含 `z_rlt`、物理空间 `ref_q`、完整 `ref_chunk` 和 `proposed_q[H,7]`。`vla` 模式的 `proposed_q` 等于 `ref_q`。提案仍须在工程机完成硬件安全投影后才可下发。

## 已验证范围与上线前检查

运行 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q src/openpi/tests/serving/rlt_pytorch_smoke_test.py src/openpi/tests/models_pytorch/pi0_pytorch_test.py src/openpi/tests/models_pytorch/rl_token_test.py src/openpi/tests/rlt/force_rlt -m 'not manual'` 检查 readout 训练/保存/server 加载/WebSocket 请求、共享前缀数值对齐、遮罩损失、E 步 TD 与回放语义。smoke test 使用小型本地 VLA 夹具，不能替代真实 Franka checkpoint 的吞吐和显存验收。此处禁用自动 pytest 插件是因为当前环境的 ROS pytest 插件依赖缺少 `lark`。

当前交付范围是 PyTorch 训练与 server 部署。`ContactPolicy`、`FrankaSafetyProjector`、`ReplayJournal` 等是离线训练和后续系统集成可复用的模块；仓库不提供新的 Franka 真机插入入口。方案第 5 节的硬件小试、在线数据采集和 A0/A1/B/C 统计依赖具体设备及数据，未在本仓库验证。


Panda 采集、训练和服务使用点号字段（LeRobot 不接受斜线）：
`observation.exterior_image`、`observation.wrist_image_left`、`observation.joint_position` /
`observation.ee_pose`、`observation.gripper_position`、`observation.gripper_image_left/right`。
训练从 `action.joint_position` / `action.ee_pose` 和 `action.gripper_position` 读取未来块，
只在模型输入时拼成内部 `actions`；无需复制采集端的 YAML。普通 DROID 数据配置保持原有接口。

真实指内 RGB 输入在 YAML 中设置 `rlt.force.input: rgb`、
`rlt.force.weights: /path/to/resnet18.pth` 和 `rlt.force.roi: [X0, Y0, X1, Y1]`
（默认全图 224×224），启用 `panda-finger-rgb-history-v1`。
客户端只采集四帧 RGB 与时间戳；GPU 服务端将基线和未缓存帧放到一个 batch 编码，
按 FingerHistory 的帧/左右指/绝对特征与两类差分顺序构造 `z_grip`。
同会话的重叠帧按时间戳缓存。四帧必须同步、唯一且新鲜；掩码不能以假数据补齐。
权重、采集的裁剪/缩放和 ROI 必须与 actor 训练一致。会话开始时需要张开且无接触的基线。
预计算输入须显式设置 `rlt.force.input: features`；RGB 输入缺少 weights 时服务会报错。
协议不要求 `actor_sha256`。机器人控制循环不执行神经编码；实际请求延迟仍需真实 GPU checkpoint 验收。
