# 普通 RLT：标签、奖励、回放和离线 RL

第一阶段的 readout 训练仍由 `openpi.training.rlt` 调度。这里实现第二阶段：
读取 data_collect 保存的轨迹和前端 episode 标注，使用冻结 VLA/readout 补齐特征，
构建 replay，训练独立的 actor 和 critic。

```text
rlt/
├── actor.py                  # 普通 RLT actor，可单独替换
├── critic.py                 # 普通 RLT twin critic，可单独替换
├── learner.py                # TD、BC、target 网络和 checkpoint
├── action_representation.py  # EE/关节动作、分位数归一化
├── dataset.py                # data_collect / LeRobot 数据适配
├── features.py               # 冻结模型特征缓存
├── episode.py                # transition 定义和 episode 窗口构建
├── replay.py                 # 每回合 Pickle 文件和 CPU ring buffer
├── build_replay.py           # 离线构建入口
├── train.py                  # actor/critic 离线训练入口
├── rewards/
│   └── terminal.py           # 可替换的 episode 奖励策略
└── force_rlt/                # 指内图像、接触控制等专用模块
```

## 标签和奖励

直接读取 dataset 根目录或历史 `audio/` 目录中的
`episode_XXXXXX.sync.json`。`success=true` 时末个有效 transition 的末步奖励为 1；
`false` 时奖励全为 0。两者都在末步终止，不再 bootstrap。
`null` 或缺失标签时跳过；重新构建时也移除该回合之前生成的 replay，避免保留旧标签。

奖励设置放在 `rewards/`。`TerminalReward` 支持 `success_reward`、`failure_reward` 和
`step_reward`；新增策略只需实现 `EpisodeRewards.generate(steps, success=...)`。
策略接收一次 episode 标签，返回按实际动作步排列的奖励数组。

## 从已有数据构建 replay

在仓库根目录、GPU 推理环境运行，不需要连接机器人：

```bash
uv run python -m openpi.rlt.build_replay \
  --dataset /path/to/data_collect/data/dataset/my_dataset \
  --config config/train/openpi_rlt.yaml \
  --checkpoint /path/to/vla_checkpoint \
  --readout /path/to/readout \
  --output /path/to/replay \
  --chunk-len 10 --stride 2
```

VLA、readout、相机协议、状态布局和归一化必须与该轨迹匹配。
CLI 使用现有普通 VLA/readout 服务加载逻辑；固定模型权重和预处理统计的摘要构成
缓存身份。窗口起点和下一状态的观测内容、任务文本和时间戳都参与缓存键，重复构建和改标会复用特征；
仅为实际需要的这些状态运行特征推理。
普通 readout 服务使用 `rlt_readout_protocol` 标识，data_collect 继续按普通 VLA 的 `actions` 执行，
不会因附带 readout 而被路由到专用接触 actor 协议。
这里的 `ref_chunk` 是从记录的观测重新推理得到的参考，不能据此还原历史执行时的随机采样参考。

默认 `--action-source command` 从 100 Hz trace 中读取每个下一观测之前的最后一个
下发目标，并检查时间覆盖。EE 使用 `target_ee_position`、`target_ee_orientation_xyzw`
和夹爪命令。关节命令要求 trace 中有 `target_joint_position`，不能用实测关节假装下发目标。

旧数据只有 LeRobot 的下一实测状态动作时，可以显式使用：

```bash
uv run python -m openpi.rlt.build_replay \
  --dataset /path/to/dataset --config config/train/openpi_rlt.yaml \
  --checkpoint /path/to/vla_checkpoint --readout /path/to/readout \
  --output /path/to/measured_replay --action-source measured
```

该选项的动作语义保存在 replay 元数据中。命令和实测动作不混入同一个构建目录。
每帧相机状态构成一个状态；N 个状态只构建 N-1 个观测到下一状态的 transition，
不会为最后一帧伪造下一图像。数据 FPS 必须匹配策略动作 FPS；明显缺帧时先对齐或切段，
不能将不等长的时间间隔当成等长 TD 步。

`--success-reward`、`--failure-reward`、`--step-reward` 可覆盖奖励默认值。
`--episode-index` 可重复提供，只重建指定回合。

## 文件和采样

```text
replay/
├── manifest.json             # 格式版本、状态/动作维度、H/E、FPS
├── episode_<identity>.pkl    # 一个 episode 的 transition 字典列表和元数据
└── features/*.pkl            # 观测对应的 z_rlt / ref_chunk 缓存
```

Pickle 使用 `HIGHEST_PROTOCOL`，不额外压缩。episode 身份包含 dataset 路径身份和索引；
重复导入或改标会原子替换同一个文件，不累加旧奖励。读取本流程生成的本地文件。
训练将 transition 放入容量有限的内存 ring buffer，均匀有放回采样 batch。

transition 包含 `z_rlt`、`proprio`、`ref_chunk`、`action_chunk`、`rewards`、
`executed_mask`、`done`、下一状态特征/机器人状态/参考块、逐步动作来源和 episode 成败。
来源为 `BASE/RL/HUMAN`，混合窗口的 `source` 为 `MIXED`。
窗口不跨回合；末端补一个对齐窗口，保证终止奖励保留。短于 E 的回合保留有效前缀 mask，
未执行部分奖励为 0，critic 和 BC 都忽略补齐动作。

## 离线训练

```bash
uv run python -m openpi.rlt.train \
  --replay /path/to/replay --output /path/to/actor_critic.pt \
  --steps 10000 --batch-size 128 --device cuda
```

默认使用物理空间绝对动作。若使用 VLA 的 EE delta 分位数统计，则一起指定表示和统计：

```bash
uv run python -m openpi.rlt.train \
  --replay /path/to/replay --output /path/to/actor_critic.pt \
  --steps 10000 --device cuda --representation ee_delta \
  --action-norm-stats /path/to/vla_checkpoint/assets/my_asset/norm_stats.json
```

参考、实际动作、下一参考使用同一变换；EE 逐臂处理六维 pose，夹爪保持绝对值，
RPY 差分做角度环绕。关节动作支持全部关节和夹爪，不采用六关节假设。
checkpoint 同时保存网络、target、优化器、动作变换和训练步数。
`--resume /path/to/actor_critic.pt` 继续训练；`ChunkLearner.propose()` 返回物理空间动作提案。

actor 和 critic 分别定义在独立文件，learner 只协调优化。
actor 使用参考输入 dropout，BC 对自主动作使用 VLA 参考、对人工步使用执行动作。
critic 使用实际执行的 E 步和 mask。TD 的 bootstrap 指数按有效步数计算，终止样本不 bootstrap。

`rlt.stage: actor_critic` 的 YAML 训练入口与上面的 CLI 都支持纯人工回合和
`RL → HUMAN → RL` 混合回合，无需额外开关。VLA/readout 只在构建 replay 时提供冻结特征，
离线 actor/critic 训练直接读取 replay，不更新 VLA/readout。
采集端的 `*.actions.jsonl` 应逐步写入 `control_source: "BASE" / "RL" / "HUMAN"`；
也识别旧字段 `source`、`vr_arm_enabled` 和布尔字段 `intervention_flag`、`human_controlled`。
`intervention_flag` 或 `human_controlled` 为 `true` 时，该步按实际人工动作训练，即使原提案来源标为 RL。
新 trace 的 `control_source` 描述实际控制来源，优先于原始 VR 按钮状态；缺少明确来源时才回退到 `vr_arm_enabled`。
每步必须具有确定来源；`MIXED` 由窗口内的逐步来源推导，不能用作单步标记。
人工接管本身不代表任务终止；混合回合在真实结束处设置 `done`，成败仍由回合标注决定。
训练日志的 `human_fraction` 表示当前 batch 有效执行步中人工步的比例。

在线服务、episode 边界切换、可选 force 模态和机器人客户端执行反馈 helper 见
[RLT 在线说明](../../../docs/rlt_online.md)。采集端使用 helper 将设备执行结果与回合成败回传。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  src/openpi/tests/rlt src/openpi/tests/serving/rlt_pytorch_smoke_test.py
```

测试覆盖真实 LeRobot v3 读回、前端标注兼容、特征缓存、改标替换、短回合和终止奖励、
动作时间对齐、EE/关节变换、actor/critic 更新以及 checkpoint 续训。合成数据测试不操作机器人。
