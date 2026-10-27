# RLT 数采、关键阶段与人工接管对照

核对日期：2026-10-27。参考为本机 `openpi-RLT` 的 `c1e40ac360185778c98cf20da2820e22d2d415e7`，当前实现为 `openpi-force` 工作树（HEAD `ee7418aa14ccceef019c3320387a15589990a9ea`）。参考目录 `/home/three/DataFromRoot/codes/grade_two/github/robot/openpi-RLT` 指向 `/data/codes/grade_two/github/robot/openpi-RLT`。

该参考版本已复制到本仓库 [third_party/openpi-RLT](../third_party/openpi-RLT/README.md)，后续可直接就地对照；复制范围和来源见[快照记录](../third_party/openpi-RLT.UPSTREAM.md)。

本文补充[实现对照](implementation_alignment_audit.md)和[PyTorch 对齐说明](pytorch_alignment.md)中的数据与运行流程。这里的结论来自源码核对；本文只更新文档，没有实现新的采集、按键或过滤功能。普通在线 RLT 的实际使用入口见 [RLT 在线说明](rlt_online.md)。独立的 `rlt.stage: force` / server `mode: force` 使用另一套 JSONL replay，不适用本文普通在线 RLT 的 600 条门槛。

## 交互实现更新（2026-10-27）

后续已实现 VR 左摇杆按下进入关键阶段、回合重置清除、推理 Y 保存 / X 丢弃以及接管/交回事件记录。
原始轨迹完整保存；离线 `build_replay` CLI 默认按 sidecar 阶段起点过滤，窗口不跨标记，
BASE/RL/HUMAN 均保留。未标阶段或成败的回合不导入；`--include-noncritical` 显式保留全任务，
两种范围使用不同输出目录。Python API 需传 `critical_phase_only=True`。
以下第 6 节关于离线 builder 不支持阶段裁剪的描述为实现前状态，当前用法见
[模块说明](../src/openpi/rlt/README.md)。普通在线 RPC 的非关键段入库及 HUMAN 后结束限制仍未改变；
本次手柄交互用于采集端保存并离线重建 replay，不代表已接通在线 RPC。

## 1. 先区分完整示教数据与训练 replay

完整任务的示教数采可以照原来的方式记录接近、抓取、移动和接触操作，供 VLA SFT、readout 训练及事后检查使用。这种采集不要求为了 actor 在线训练而逐次按 `c`。是否保留某段原始数据，与它是否进入 actor–critic 的训练 replay，是两个决定。

如果同一批完整示教还要作为“只学习关键接触段”的 RL warmup 数据，就必须有办法确定关键段范围：在采集时记录阶段事件、事后标注并切段，或者整个回合从预设的关键段起点开始。只有完整轨迹和一次成败标签，不能自动恢复关键阶段的起止位置；actor 在线训练本身也不会完成这项筛选。

| 数据用途 | 采集范围 | 关键阶段标记的要求 | 如何用于学习 |
| --- | --- | --- | --- |
| VLA 示教/SFT、第一阶段 readout | 可以保留完整任务 | 不要求使用在线 RL 的 `c` 标记 | 按训练配置读取示教数据，不计入在线 warmup 的 600 条 |
| 接触 RL 的 warmup | 原始轨迹可完整保存，训练 replay 只取关键阶段 | 从全任务起点运行时需指定边界；从关键起点运行时默认全回合属于关键段 | actor 尚未就绪，通常执行 VLA，也可包含人工动作；提交后的有效 replay 条目才计数 |
| 接触 RL 的 online 交互 | 同样保留关键段的自主与人工动作 | 沿用 warmup 的阶段定义 | 执行已发布 actor，收集新数据并追加更新预算 |

以上是与参考仓库接触 RL 流程一致的数据范围。当前普通 RLT 尚未完整实现这套筛选，差异见第 3、6 节。

## 2. 原仓库怎样进入关键阶段

`critical_phase` 表示任务中需要精细接触操作的阶段，由任务配置或操作者指定。原仓库没有根据接触力、图像或距离自动识别这个阶段。

| `runtime.env_driver.task_mode` | 起点和阶段状态 | 操作者的操作 | replay 范围 |
| --- | --- | --- | --- |
| `full_task` | 从完整任务起点开始，`critical_started=False` | 按 `o` 开始回合，观察到关键阶段边界时按 `c` | 前缀标为 `drop_transition=True`；进入关键阶段后的连续片段才构建训练窗口 |
| `critical_phase` | 使用 `critical_phase_reset_action` 作为复位目标，回合初始化即 `critical_started=True` | 按 `o` 开始，不需要再按 `c` | 回合从关键阶段开始，符合窗口条件的数据进入 replay |

训练键盘中的 `c` 调用 ROS `/enter_critical_phase`，将当前回合的 `critical_started` 设为 `True`；重复按 `c` 不会退出关键阶段。数据执行循环在每个 chunk 开始时读取阶段标志，因此在 chunk 执行中按 `c`，当前 chunk 仍按原来的阶段分类，后续 chunk 才按关键阶段采集。新回合按 `task_mode` 重新初始化标志。

当前参考任务配置 `rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml` 使用 `task_mode: critical_phase`，所以通常看不到运行到一半再按 `c` 的步骤。它省略了人工切换，仍然有明确的关键阶段起点。

原仓库会先保存带有 `drop_transition` 标记的原始回合，再在构建 replay 时跳过被排除的 chunk，并切开连续片段，避免训练窗口跨过被排除的区间。因此“非关键段不进 replay”不等于“原始日志里没有这段数据”。

## 3. 两边当前实现的差异

| 项目 | 参考 `openpi-RLT` 的 ROS 在线链路 | 当前 `openpi-force` 普通在线 RLT |
| --- | --- | --- |
| 指定关键阶段 | `task_mode` 与 `critical_started`；`full_task` 用 `c` 进入 | `critical_phase_only: true` 时每次 `infer` 提供 `in_critical_phase`；默认此配置为 `false` |
| 非关键阶段动作 | 使用 VLA reference | 开启阶段限制后使用 VLA reference |
| 非关键阶段入库 | 标记 `drop_transition=True`，构建 replay 时排除 | `observe` 仍追加 draft，`end_episode` 将本轮 drafts 一起提交 |
| warmup 计数 | learner 统计训练 replay 的大小；正常由此 ROS 链路构建时只含关键段 | learner 统计整个 buffer 的大小，不能自动区分关键/非关键条目 |
| `c` / `t` 操作入口 | 有 ROS 键盘客户端、阶段信号与遥操作切换 | 当前 RLT helper 提供 RPC，不包含原仓库的键盘/ROS 控制实现 |
| 人工动作来源 | 逐控制步记录 `HUMAN`，混合窗口为 `MIXED` | 在线一次 `observe` 的所有有效动作使用同一个来源，可显式传 `HUMAN` |
| 人工交回策略 | 支持同回合接管、恢复策略和重新规划 | `HUMAN` 反馈后必须结束本回合，不能直接继续 `infer` |
| 离线混合数据 | 可记录并构建逐步来源的回放 | 离线导入支持 `RL → HUMAN → RL`；这不表示在线 RPC 已支持同回合恢复 |

当前 `critical_phase_only` 只控制动作来源，没有把阶段信息带到 pending transition 和入库筛选中。单独把它改成 `true`，不能使 warmup 只统计关键段。原仓库的 learner 也不会重新识别外部导入 replay 的阶段；“只含关键段”依赖上游正确筛选。

## 4. warmup 的 600 条到底统计什么

两边均统计训练 buffer 中的 **chunk transition 条目数**，不是摄像头帧数、单步动作数、回合数、按键次数或梯度更新次数。warmup 和 online 使用同一个训练 replay，关键阶段的 `BASE`、`RL` 和 `HUMAN` 数据都可以参与计数；不存在一个天然只装 warmup 数据的独立 buffer。

参考 AgileX Ethernet 任务设置 `warmup_min_size: 600`、`warmup_post_collect_updates: 20000`；参考通用配置的 `warmup_min_size` 是 1000，不能把 600 当作所有参考配置的默认值。当前服务默认 `warmup.min_transitions: 600`、`warmup.updates: 20000`。

原仓库正常在线采集只将关键段构成的 transition 放入 replay，因此达到 600 条意味着有 600 条关键段训练窗口。当前服务则会把本轮非关键段也提交；即使这些动作由 VLA 执行，也会增加 warmup 计数。

例如，起初已有 590 条关键段 transition；新一轮按同一统计口径产生 20 条候选条目，其中只有 5 条属于关键段。参考流程过滤后为 595 条，继续等待；当前未过滤的在线流程会变成 610 条，并触发初始训练预算。这说明阶段过滤会实际改变开始训练的时机。

窗口规则也会影响条目数。原仓库支持 chunk 窗口及 `step_trace_stride > 0` 的密集窗口；当前在线服务每次有效 `observe` 产生一条 draft，离线 builder 则按 `stride` 滑窗并保留末端窗口。相同机器人运行时间不保证产生相同数量的 transition，比较数据预算时还应报告实际控制步数、交互时间和窗口设置。

当前服务从空 buffer、未训练 checkpoint 开始时，顺序如下：

1. `online_rl` 启动后台 learner，warmup 回合由 VLA 产生动作；客户端返回实际执行反馈。
2. 回合中 draft 尚未提交，不增加可采样 buffer 的大小。`end_episode(success=True/False)` 生成奖励并提交整轮；`abort_episode` 丢弃未提交数据。
3. 达到数据门槛后自动唤醒 learner，执行初始更新预算，不需要另发“开始训练”命令。未达到门槛时 `global_step` 保持 0；恢复训练时它保留 checkpoint 中的进度。
4. 数据达到 600 条不表示 actor 已可执行；默认还要完成 20000 次 learner 更新并发布符合条件的 actor 快照。`actor_update_period: 2` 时，其中约 10000 次更新 actor。
5. 满足就绪条件后，在下一次 `begin_episode` 切到 online；数据够而快照未就绪时返回 `warmup_wait_online`。online 后每新增条目默认追加 5 次 learner 更新预算，预算用完后等待新数据。

“只用关键接触阶段的数据训练”限定的是样本范围。后台更新可以在机器人不接触、回合已结束时继续进行，不要求物理接触与梯度更新同时发生。启动时加载已有 replay，也可能直接满足数据门槛。

## 5. 人工接管怎样采集和训练

原仓库的 `c` 与 `t` 分别控制任务阶段和人工控制权。`c` 进入关键阶段；`t` 在策略控制和人工遥操作之间切换。warmup 期间策略控制是 VLA；online 且 actor 就绪、处于关键阶段并选择 actor 时，策略控制才使用 actor。按 `c` 本身不会跳过 warmup。

接管时停止策略发动作，`HumanActionRecorder` 从遥操作关节命令流中缓存最新动作，执行循环按控制周期采样。交回策略时有恢复等待和运行状态清理，并从当前状态重新规划。`t` 不会自动结束回合，也不表示成功或失败；仍由 `s` / `f` 标记任务结果。非关键阶段的人工动作同样被排除在训练 replay 之外。

人工数据需要保留以下语义：

| 字段/内容 | 用途 |
| --- | --- |
| 实际执行动作与逐步来源 | 区分 `BASE/RL/HUMAN`；窗口包含不同来源时推导 `MIXED` |
| 同一观测的 VLA `ref_chunk`、RL token 和机器人状态 | 作为 actor 的参考与条件输入，人工动作不替换 VLA reference 字段 |
| 下一观测、奖励、终止状态 | 用实际执行动作构建 critic 的 TD 目标；任务结果由回合标注决定 |
| 阶段范围与时间戳 | 过滤非关键段，保持动作、观测和连续训练窗口的时序一致 |

两边 actor 的 BC 都对人工步使用实际人工动作作为目标，对自主策略步使用 VLA reference 作为目标；critic 使用实际执行动作。原仓库还支持可选分层采样，为人工干预池分配采样比例；当前普通 replay 使用均匀有放回采样。

当前在线接口可用 `observe(..., source="HUMAN")` 回传一次人工执行反馈，然后必须 `end_episode`，由调用方明确给成败标签。该边界限制与原仓库同回合 `RL → HUMAN → RL` 的行为不同；不要把人为切段自动等同于一次任务失败。需要连续接管并恢复策略时，还需补齐在线协议和执行端控制。

## 6. 当前 data_collect 离线数据能复用到哪里

当前普通离线链路为：`data_collect / LeRobot` 轨迹及 sidecar → 冻结 VLA/readout 提取特征 → `build_replay` → actor–critic 训练。操作命令和字段详见[模块说明](../src/openpi/rlt/README.md)。

`episode_XXXXXX.sync.json` 提供回合成败，`*.actions.jsonl` 提供实际下发目标及逐步控制来源。默认 `--action-source command` 使用执行 trace，与下一观测前已下发的目标对齐；不能把 server 提案或实测状态直接当作实际下发动作。旧数据的 `--action-source measured` 是明确选择的另一种动作语义，构建目录不可混用。

来源字段优先记录实际控制来源 `control_source: BASE/RL/HUMAN`；导入器还兼容 `source`、`vr_arm_enabled`、`intervention_flag` 和 `human_controlled`。它能保留整轮人工示教及 `RL → HUMAN → RL` 的逐步来源。`success=null` 或缺少标签的回合不构建训练 replay，重新构建时移除该回合旧 replay。

**当前离线 builder 没有关键阶段起止的导入/裁剪接口。** 它对选中的已标注回合整体构建窗口；`--episode-index` 只能选回合，`--stride` 只控制窗口步幅。只有成败、来源或“正在人工控制”标记，不能据此推断关键接触阶段。若要从完整示教构建仅关键段的 warmup replay，需要先补标并按连续区间切段，同时保留下一观测与真实任务终止信息；当前命令不会自动完成这一步。

离线构建中的 `ref_chunk` 来自冻结 VLA 对记录观测的重新推理，不保证等于历史执行时随机采样的参考。复用数据时应记录模型、readout、预处理和动作统计的身份；数据与特征一致性之外，还需核对阶段筛选口径。

## 7. 与参考流程对齐时的数采要求（待实现）

完整示教数采继续保留全任务。接触 RL 从 warmup 第一轮起就采用一致的关键阶段定义：`full_task` 显式指定进入关键阶段；从关键起点开始时整轮默认为关键段。原始记录保留阶段、控制来源和成败信息，训练 replay 只选有效关键段。人工接管与关键阶段切换独立，并支持交回策略后继续同一回合。

实现时应能核对以下结果；这些是后续验收要求，本次文档更新没有宣称它们已经通过：

- 非关键段无论由 VLA 还是人工执行，都不增加接触 RL 的 buffer 大小和新增数据更新预算。
- 关键段的 VLA/人工 warmup 数据在回合标注并提交后才计数；abort 的未提交回合不增加计数。
- 原有 599 条时，新增非关键段仍不触发训练；新增并提交有效关键段使条目数达到 600 时自动开始更新。
- 阶段边界在采集端有明确的生效 chunk/时间戳，replay 窗口不跨越排除区间，离线重建可复现相同范围。
- 人工接管及恢复保留逐步来源、实际动作和 VLA reference；接管本身不改写任务成败、不制造真实任务终止。
- 日志分别报告原始控制步、排除步、有效关键段步、提交 transition 数、人工步比例与 learner 更新数，便于比较两边数据预算。

## 8. 源码定位

参考路径以下表为准，相对于本仓库 `third_party/openpi-RLT/`；与上述原始参考目录的对应文件内容一致。当前文件使用仓库内链接。

| 核对项 | 参考仓库文件/符号 | 当前仓库文件/符号 |
| --- | --- | --- |
| 阶段和人工按键 | `rlt_online_rl/keyboard_toggle_teleop_record_reward_isolation.py`：`main`、`toggle_teleop` | [RLTClientPolicy](../src/openpi-client/openpi_client/rlt_client_policy.py)：`infer`、`observe`、`end_episode` |
| 阶段状态与接管 | `rlt_online_rl/train_deploy_alignment/{manual_signal_bridge,pika_sync_ros}.py`：`reset_episode_state`、`enter_critical_phase`、`HumanInterventionState`、`HumanActionRecorder` | [OnlineRLTPolicy](../src/openpi/serving/rlt_policy.py)：`_infer`、`_observe`、`_end` |
| 关键段过滤和 replay 构建 | `rlt_online_rl/train_deploy_alignment/pika_sync_ros.py`：`execute_chunk`；`rlt_online_rl/src/rlt_online_rl/inference.py`：`_collect_replay_segments`、`_build_episode_replay` | [dataset.py](../src/openpi/rlt/dataset.py)：`import_dataset`；[episode.py](../src/openpi/rlt/episode.py)：`build_episode_transitions` |
| warmup 与更新预算 | `rlt_online_rl/src/rlt_online_rl/trainer.py`：`train_once`、`_desired_total_updates` | [online.py](../src/openpi/rlt/online.py)：`commit_episode`、`_desired_updates`、`train_once` |
| 人工 BC 目标和采样 | `rlt_online_rl/src/rlt_online_rl/{trainer,replay}.py`：`bc_target`、`human_intervention_pool` | [learner.py](../src/openpi/rlt/learner.py)：`update`；[replay.py](../src/openpi/rlt/replay.py)：`ReplayBuffer.sample` |
| 默认与任务参数 | `rlt_online_rl/configs/base/online_rl.base.yaml`、`rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml` | [server YAML](../config/server/openpi_rlt.yaml)、[settings.py](../src/openpi/rlt/settings.py) |
