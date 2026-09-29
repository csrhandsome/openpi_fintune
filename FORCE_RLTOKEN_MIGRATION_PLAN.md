# Force RL Token 迁移方案（先做插入任务）

## 1. 目标与方法边界

本方案把 `../openpi-RLT` 的**真实 RL-token readout + 小型在线 actor–critic**迁到本仓库，让 VLA 完成抓取、粗定位和接近，让接触段 actor 利用两个软手指内 RGB 相机的形变特征修正短动作块。这里的“force”仅指与接触力相关的**视觉形变线索**；不标定牛顿/牛米，不回归力值，不把形变特征称为已测量力。

首个任务只做毫米级位置/姿态误差下的插入、卡滞与重试。断电灯泡式旋拧是第二阶段任务，需另行处理连续旋转、轴向推进和滑移，不从插入结果推断其性能。

主方法必须同时满足：VLA 内部表征产生经单独训练的 RL token；VLA 及 readout 在接触 RL 期间冻结；actor **直接生成**受 VLA 参考约束的短动作块；critic 从实际执行动作与独立任务结果学习。若读不到内部表征或 readout 不能训练，只能称“RLToken 启发的参考动作适配”，不能沿用主方法名称。

## 2. 代码现状与迁移边界

| 现有代码 | 查到的接口/限制 | 迁移决定 |
| --- | --- | --- |
| `src/openpi/models/pi0.py`、`src/openpi/training/config.py` | 当前普通 Pi0 没有 `extract_prefix_embeddings` 等 RLT 接口；`pi0_base_joint_ros_30hz` 是 16 步、30 Hz、模型 32 维填充的 Franka 配置。 | 以**不读取指内图像**的 Pi0 checkpoint 为主实验 VLA 候选，先核对实际 checkpoint、相机输入、成功接近率和归一化资产。必要时训练仅用指定外部视角的 VLA；所有组冻结同一 checkpoint。 |
| `src/openpi/models/pi0_force.py`、`src/openpi/policies/forcevla_policy.py` | 已有指内图像的 CBAM + GRU + MoE 融合，直接影响 VLA 动作；没有所要求的独立 ResNet 接触分支。 | **不作为 B/C 主对照底座**，否则 B 仍有指内信息或删输入会改动 VLA。保留为以后独立的端到端视觉接触模型比较。 |
| `vla-scripts/realworld/franka_joint_main_rtc.py` | 现接入外部/腕部相机与关节/夹爪状态，输出前 7 维绝对关节目标及第 8 维夹爪命令，按 30 Hz 执行；没有指内相机接入、自动接触交接和回合奖励。`max_joint_delta=0`、`joint_velocity_limit=0` 默认代表未启用相应软件限制。 | 复用其相机请求、RTC 与 Franka 控制经验；新增接触控制器、双指相机采集、时间戳、硬联锁和日志。在线 RL 前把全部限制配置成经过空载验证的**非零物理值**。 |
| `../openpi-RLT/src/openpi/models/rl_token.py`、`scripts/train_rlt.py`、`scripts/serve_rlt_policy.py` | readout 对 VLA 的 post-transformer prefix 做交叉注意力压缩与重建；服务返回 `z_rl`、`proprio`、`ref_chunk`。服务中 `PROPRIO_DIM=7`、`CHUNK_LEN=50`、`ACTION_DIM=7` 为硬编码。 | 移植模型/训练逻辑，服务输出长度和维度改从选定 checkpoint/运行配置推导；单独保存 readout，并校验 VLA hash 与 token 形状。 |
| `../openpi-RLT/rlt_online_rl/src/rlt_online_rl/{networks,trainer,replay,inference,action_representation,config}.py` | JAX 固定方差 chunk actor、twin critic、target 网络、BC/Q/参考平滑正则、回放和 actor 服务；仅处理 `z_rl, proprio, ref_chunk`，没有双指观测。原动作表示和关节平滑项针对另一台机器人。 | 有选择地迁移算法核心与服务协议；所有 state/next-state、网络和 replay 加 `z_grip`，重写 Franka 动作适配与安全投影，不照搬 AgileX ROS 适配器及其物理参数。 |

`../openpi-RLT` 仅作只读实现参考；本方案的实现全部落在当前仓库。当前工作树已经有 Franka 30 Hz 相关改动，迁移时应在这些文件上小范围增量修改，避免覆盖现有实验。

## 3. 首版固定的数据流

```text
外部相机 + 指令 + q/夹爪状态 → 冻结 VLA → z_rlt + 16 步 VLA 参考
                                        ↓                  ↓
左/右指内 RGB → 同权 ResNet-18 → 左/右独立特征 + 4 帧历史/差分
                                        ↓
             (z_rlt, z_grip, q_t, ref_chunk) → actor → 短关节目标块
                    (同一状态, 实际执行块) → twin critic
```

- **VLA 视角**：主实验的 VLA 只能用选定的外部视角、指令和机器人状态。当前 Franka 配置/脚本还提供腕部视角；如果现有 checkpoint 依赖腕部视角，就必须在 A/B/C 中同样保留并如实描述，或先训练一个仅用外部视角的 checkpoint。两台指内相机始终不进入主实验 VLA/readout 的输入。
- **RL token**：沿 RLT 参考实现在普通 `Pi0` 上暴露 prefix-only readout；固定插入指令时先按参考实现只压缩 image prefix，状态单独输入 actor。训练时应使用有效 image token mask，并保证训练/推理的 token 选择、mask 和预处理完全一致；参考 `train_rlt.py` 当前将 `None` 传给重建 loss，迁移时须验证遮罩行为，避免把被屏蔽的相机 token 当有效数据。保留 readout 的重建损失，不从最终成败反向更新 VLA/readout。
- **形变特征**：初版用一个共享权重 ResNet-18 分别编码左右相机；每路保持单独向量，按左/右顺序拼接，不能平均掉接触方向。每个抓取回合记录无接触参考帧，取最近 4 个同步帧的特征及相对该参考帧/上一帧的变化量；时间戳、有效位一起记录。先冻结 encoder。若离线可观测性小试失败，再只用训练集微调或训练自监督 encoder，并作为独立消融记录；不训练力回归头。若采用 PyTorch `torchvision` ResNet，明确补齐与当前 `torch==2.7.1` 匹配的依赖和权重来源；encoder 输出通过明确的 NumPy/JAX 边界进入 RL 服务。
- **接触交接**：外部视觉与实测 TCP 位姿共同判定“进入接触邻域”；阈值、方向和连续确认帧数在收集主实验数据前写入任务配置，并对 A/B/C 共用。交接记录实测位姿、抓取状态和起点误差，按起点分层/配对评估。接触段继续低频请求 VLA 参考；每次回包标记观测时间、生成时间和起始 TCP。参考过期时禁止直接执行旧绝对目标，应按当前实测 TCP 重新锚定；超时或 IK 失败则保持/安全撤退并记录中止。
- **动作接口（首版建议，必须在 RL 前锁定）**：利用现有 Franka 接口，actor 输出 7 维**绝对关节位置短块**，夹爪保持交接时的闭合命令；不让 actor 改抓握力或开夹爪。候选为 30 Hz 底层执行、`H=4` 步预测、每次仅执行前 `E=2` 步后重观测（15 Hz 接触重规划）。这是待无学习小试确认的工程建议，若相机/推理延迟达不到 2 个控制周期，统一调整 `H/E/频率` 后冻结所有组设置。模型输出维度 7 与 VLA 的物理 8 维、内部填充 32 维明确分离。
- **TCP 与安全投影**：固定 Franka 基座坐标 `O`、工具 TCP 标定、姿态表示和右手系；用 FK 将参考/候选关节块转成 `O_T_TCP` 检查每轴 TCP 平移、转角、速度、加速度及工作区边界，另检查关节位置/速度/加速度。旧参考的 TCP 相对轨迹以当前实测 TCP 为新起点，经 IK（种子为当前实测 `q_t`）生成新关节参考；无法连续求解即拒绝该块。网络输出先限幅到参考邻域，再由策略外安全层投影/拒绝。critic 和回放记录**实际下发并执行**的目标以及裁剪/拒绝标记，不能把未经投影的提案当实际动作。所有轴向上限及超时值由硬件空载测试确定，不在此文档猜测数值。

不在最小实验中引入 latent-noise steering。只有确认 VLA 的 diffusion/flow 初始噪声可控且插入任务可 steer，才考虑真正的 DSRL 式对照；普通关节残差不称 DSRL。

## 4. 具体模块与协议改造

建议按以下路径落地，提交时可微调文件名，但接口字段与测试应保持稳定。

| 当前仓库目标 | 改动 |
| --- | --- |
| `src/openpi/models/pi0.py` | 引入 RLT 的 `extract_prefix_embeddings`、必要时共享前缀 KV cache 的推理路径；以原 `sample_actions` 结果作数值对齐。不得改变已冻结 VLA 权重或原始 A 组动作。 |
| `src/openpi/models/rl_token.py`、`scripts/train_rlt.py`、`scripts/serve_rlt_policy.py` | 移植 readout、独立训练与推理，输出 `z_rlt` 和**物理空间** VLA `ref_chunk`；动态处理 16 步/7 关节/夹爪维度；在元数据中保存 VLA/readout checkpoint 与相机输入协议版本。 |
| `src/openpi/force_rlt/tactile_encoder.py`、`observation.py` | 两路图像采集后的裁剪、颜色顺序、无接触参考、4 帧 ring buffer、共享 ResNet 编码、时间同步和缺帧标记；仅接触端使用。 |
| `src/openpi/force_rlt/{networks,learner,replay,action_adapter}.py` | 从 RLT runtime 迁移 actor/twin critic/target、训练调度和 journal；显式增加 `z_grip` 与 `next_z_grip`，critic 输入当前参考块及实际动作块；采用 Franka 7 维归一化与执行转换。不要沿用原代码只处理前六轴的 `delta_weight` 规则。 |
| `vla-scripts/realworld/franka_force_rlt_insert.py` 与任务配置 | 从现有 Franka RTC 脚本抽出/复用控制与 VLA 客户端，新增自动交接、双指相机、actor 查询、奖励/终止、重置、在线记录、人工接管隔离和独立安全层。不要把 `pika_sync_ros.py` 直接当 Franka 驱动。 |
| `tests`/对应 `*_test.py` | 测 token/动作形状与对齐、双相机次序/同步、参考重锚定、裁剪与过期回退、终止回放边界、接管奖励隔离。 |

服务消息和 replay 至少保存以下字段；原始 RGB 可放独立压缩记录并以帧 ID 引用，训练可缓存冻结 encoder 的特征，但必须保存 encoder hash 与预处理版本以便重算：

```text
observation: episode_id, step_id, monotonic_ts, external_frame_id,
             finger_left/right_frame_id + timestamps + valid_mask,
             q_t[7], gripper_state, O_T_TCP, contact_phase
features:    z_rlt[D_rlt], z_grip_left/right/history[D_g],
             ref_q[H,7], ref_obs_ts, ref_generation_ts, ref_anchor_TCP
transition:  (features, q_t, ref_q), proposed_q[H,7], executed_q[E,7],
             reward[E], done, next_features, next_q, next_ref_q,
             source={BASE,RL,HUMAN}, actor_version, safety_flags,
             independent_success, intervention_flag
```

训练使用短块 TD 目标时必须区分 `H`（策略预测）和 `E`（真实执行）：actor 保持 `H` 步输出与参考块 BC 约束，critic 只估计已执行前缀 `a[:E]`，TD 折扣和奖励也只跨 `E` 个控制步；target actor 同样只取前 `E` 步。回合中途结束的步设置有效 mask，不能把未执行的预测步填成零动作然后照常 bootstrap。B 组使用相同网络形状的常量零指内特征、重新初始化并训练自己的 actor/critic；采集和评估时指内相机不用于策略，软手指的被动顺应性保持一致。

## 5. 实施顺序与通过条件

1. **封存任务与底座**：选定插入件、孔位、成功判定、自动/人工重置流程、VLA checkpoint、视角、数据集、资产 hash、允许的在线回合/机器人小时。先用同一个 VLA 测其是否稳定到达接触邻域。若原 checkpoint 用指内图像或无法稳定接近，先解决底座，不进入在线 RL。
2. **无学习控制小试**：把双指相机接入并同步到实测 `q/TCP`，验证坐标系、夹爪不误开、`H/E` 延迟、参考重锚定、硬限位、急停、超时、重置。只执行受限 VLA 块和手写小残差，覆盖无接触、平移/转角偏差、卡滞、滑移；失败即停在此阶段。安全层应在 actor 服务不可达、指内帧过期、相机掉帧或 VLA 参考过期时独立接管。
3. **可观测性与 readout**：按物体、抓取位置、采集日划分训练/测试；检查 `z_grip`/历史是否在仅用机器人状态之外区分对控制有用的接触状态。移植并训练 RL-token readout，验证重建损失、token 非塌缩、冻结 VLA 推理与原 checkpoint 动作一致、`z_rlt` 与参考来自同一观测。若跨条件无增益，先修相机安装、光照/ROI 或表征。
4. **离线链路验收**：用记录的插入序列生成 transition，回放到 learner 与 actor 服务；检查左右特征、next-state、奖励、terminal mask、真实执行动作、接管后的数据分段和 checkpoint 恢复。先在离线/影子模式运行 actor，统计推理和安全投影触发频率。
5. **受控在线插入**：从匹配的接触起点收集固定预算的 A0/B/C；B、C 分别训练，固定 VLA/readout、actor 容量、网络训练步骤、参考刷新、安全参数和探索预算。奖励以独立判定成功为终点，辅以统一的时间/重试成本，卡滞、滑移、越界按预先写定规则终止或惩罚。安全联锁的腕/关节力矩信号（若硬件有）只用于停机，不进 actor/critic 观测或标签。人工接管后成功不得回传给接管前自主动作；接管数据需单列来源和使用规则。
6. **端到端与外推**：A1/B/C 再从完整抓取起点测试；训练件和未见物体/孔位、抓握位置、夹爪状态与不同采集日分别报告成功率及置信区间、完成时间、重试、在线交互/机器人小时、卡滞和安全中止。插入结果成立后才开启断电旋拧件研究。

## 6. 主实验与解释准则

| 组 | 定义 | 用途 |
| --- | --- | --- |
| A0 | 同接触起点，冻结 VLA 直接执行接触段，使用相同交接与安全层。 | 测接触段原始 VLA。 |
| A1 | 同任务分布，从抓取开始完整冻结 VLA rollout。 | 测端到端底座；不与匹配起点实验混算。 |
| B | 冻结同一 VLA/readout + 新训练小 actor–critic，指内策略输入恒零。 | `B−A0` 估计小 RL 适配贡献。 |
| C | 与 B 相同配置/预算 + 双指 ResNet 形变特征，重新训练 actor–critic。 | `C−B` 估计主动形变观测贡献。 |

预先固定主指标、最小有意义提升、随机种子/重复批次、样本预算和停止规则。核心消融依次做单帧与 4 帧/差分、冻结与微调 encoder、持续刷新与交接缓存参考、真实 RL-token 与移除 token；后者需重训 actor，不可只在 C 的已训练模型上置零。动作空间比较放在主实验稳定后。

只有 C 在同预算下稳定优于 B 与 A0，且在未见条件/独立采集日保持收益、没有增加安全中止，才能声称“未标定的指内视觉形变帮助 VLA 末端接触操作”。若 C 只优于 A0 而不优于 B，收益主要来自 RL 适配；若只在训练件有效，则收窄为本硬件/任务可行性。没有同硬件标定 wrench 对照，不声称优于力/力矩传感器。

## 7. 迁移完成的验收清单

- 一次 VLA 请求能产生时间对齐的 `z_rlt` 和物理空间参考块；VLA/readout/encoder/actor 各自可独立加载，版本与 hash 写入回放。
- 主实验 VLA/readout 完全不读取指内图像；B/C 仅在主动形变观测不同，且各自重训小策略。
- 可重放同一接触轨迹，复原双指特征、实际执行块、成功/终止和每次安全干预；所有数值维度由配置校验，无 RLT 原硬编码残留。
- 交接、动作投影、旧参考重锚定、超时与人工接管已在空载/影子模式通过；安全层可在 actor/VLA 失败时独立停止。
- 插入实验可复现实验起点、预算、成功判定和 A0/A1/B/C 统计；再决定是否开展旋拧。
