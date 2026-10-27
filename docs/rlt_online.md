# RLT 在线训练与推理

普通 VLA 使用 `config/server/openpi.yaml`。RLT 使用
`config/server/openpi_rlt.yaml`，通过 `train_config` 引用原训练 YAML 的模型、
相机、动作布局和预处理。所有相对路径从仓库根目录解析；CLI 的 checkpoint、readout、
device、host、port、record 覆盖 inference YAML 中的对应值。

```bash
uv run scripts/server.py --config config/server/openpi.yaml
uv run scripts/server.py --config config/server/openpi_rlt.yaml
```

RLT 启动前设置 VLA checkpoint 和 readout 路径。VLA checkpoint 必须包含
`model.safetensors` 与归一化资产，readout 包含 `readout.safetensors` 与 `manifest.json`。
原来的 `--config config/train/openpi_rlt.yaml` 仍可启动只输出 VLA 动作与 readout 特征的服务。

## 运行模式和控制权

`rlt.mode: online_rl` 冻结 VLA/readout，由后台 PyTorch learner 更新小 actor–critic。
推理 actor 是与训练网络分开的冻结副本，按 `publish_interval` 发布参数；warmup 达标时
立即额外发布一次。推理不与 learner 共用正在更新的 actor。

| 阶段 | 动作来源 | 条件 |
| --- | --- | --- |
| warmup | VLA 的 `ref_chunk`，source=BASE | 已标注 replay 少于 600 条 chunk transition |
| warmup_wait_online | 等待，尚未接受新 episode | 数据够了，但 20000 次 learner 更新或对应 actor 快照尚未就绪 |
| online | actor 的 refined chunk，source=RL | learner 达标且推理 actor 已加载达标后的快照 |

阶段只在 `begin_episode` 时确定，当前 episode 不因后台进度改变控制来源。
开始请求返回 `accepted: false` 时，机器人客户端等待并重试，不能执行上一轮的动作。
默认整个 episode 使用选定策略。设置 `critical_phase_only: true` 后，客户端必须在每次
推理请求中提供 `in_critical_phase: bool`，非关键阶段继续执行 VLA。
`begin_episode(..., policy_mode="base")` 可以选择本轮始终执行 VLA。

这里的关键阶段开关目前只限制动作来源，不过滤采集数据。即使
`in_critical_phase=False`，有效 `observe` 仍生成 draft，标注结束后也会入库并计入
warmup；默认 600 条统计整个训练 buffer 的 chunk transition 数。它与参考
`openpi-RLT` 的“非关键段不进入 replay”存在差异。完整示教数采、RL warmup、
关键阶段及人工接管的逐项对照见 [数采对照](rlt_collection_alignment.md)。

20000 表示 learner 更新数；`actor_update_period: 2` 时，actor 更新约 10000 次。
收集完成 warmup 后，每新增 transition 增加 `grad_updates_per_cycle` 次更新预算，
没有新数据时停止追加更新。`freeze_after_warmup: true` 则达标后停止学习。
checkpoint 保存网络、target、优化器、动作变换、训练步数及在线更新预算进度；
设置 `rlt.actor_checkpoint` 可恢复。已有输出 checkpoint 时须明确恢复，防止覆盖已有训练。

`rlt.mode: eval` 要求 `actor_checkpoint`，使用固定 actor 执行，不启动 learner，
不进行 warmup。建议 `deterministic: true`、`collect.enabled: false`。
开启 eval 收集时仍可保存带标注的轨迹，但不更新权重。
普通 VLA 的评估直接使用 `inference/openpi.yaml`。

## 可选收集和 episode 标注

`collect.enabled: true` 时，客户端必须回传实际执行反馈。当前 episode 先保存在内存，
收到一次成功/失败标注后才写入 replay，并进入可采样 buffer。默认成功末个有效动作奖励 1，
失败奖励 0，其余动作奖励 0；两者末个 transition 都 `done=true`，不 bootstrap。
未标注或 `abort_episode` 的回合不进入训练 buffer。

每次最多执行 `chunk_len` 个动作。反馈记录机器人确认执行的物理空间绝对目标，
包括执行端裁剪后的目标；不能把 server 的提案当作实际执行，也不能把实测状态当作下发动作。
buffer 保存 `z_rlt`、物理机器人状态、参考 chunk、执行动作、执行 mask、奖励、下一状态、
动作来源和回合成败。少执行的尾部使用 mask，最后有效动作获得终止奖励。
客户端执行频率须匹配训练配置的 `action_fps`，E 步必须是连续、等间隔的控制步。

`collect.enabled: false` 时不用回传执行反馈；online_rl 需要已有且匹配模型的 replay，
并仍须满足 warmup 条件。关闭收集不会绕过切换门槛。eval 可完全不使用 replay。

客户端使用 `openpi-client` 中的 helper：

```python
from uuid import uuid4
from openpi_client.rlt_client_policy import RLTClientPolicy

client = RLTClientPolicy(host="SERVER_IP", port=8001)
client.begin_episode(str(uuid4()), timeout=1800)  # 等待完成后再启动机器人任务
response = client.infer(read_robot_observation())
while True:
    # 由机器人执行端实现：只执行前 E 步，返回确认下发的目标、下一观测和是否结束。
    executed, next_obs, finished = execute_chunk(
        response["actions"][:response["execution_horizon"]]
    )
    client.observe(executed, next_obs)
    if finished or len(executed) < response["execution_horizon"]:
        break
    # observe 已生成下一状态的 VLA 特征与参考；复用它，避免重新随机采样参考。
    response = client.infer()
client.end_episode(success=episode_success)  # 每个 episode 只标一次
client.close()
```

默认 Franka 状态从 `observation.ee_pose`/`observation.joint_position` 和
`observation.gripper_position` 组合；外部相机、腕相机和 prompt 沿用普通 VLA 协议。
自定义状态可传 `observation.proprio`，须匹配 `proprio_dim`。
每个 server 同时运行一个机器人 episode。`episode_id` 必须唯一，反馈的 `step_id`
须对应尚未确认的动作。人工接管反馈使用 `source="HUMAN"`，随后结束回合，避免跨接管边界。
此在线接口尚不支持原仓库同回合接管再恢复策略；实际遥操作切换由执行端负责，
helper 不提供 `c` / `t` 按键。人工接管本身不代表任务失败，成败仍需显式标注。
动作提案由机器人现有执行接口下发；执行端继续负责设备约束、急停和目标裁剪。

底层仍是原 WebSocket，消息为 `{rlt_event: ..., episode_id: ..., ...}`：

| rlt_event | 额外字段 | 返回 |
| --- | --- | --- |
| status | 无 | buffer 数量、global_step、actor_version、published_step、ready_for_online、learner_error |
| begin_episode | 唯一 episode_id，policy_mode=actor/base | accepted、phase |
| infer | 首次提供 observation；关键阶段限制开启时提供 in_critical_phase | actions、ref_chunk、source、step_id、actor_version、execution_horizon |
| observe | step_id、executed_actions[N,D]、next_observation，可选 source | 执行反馈确认；缓存下一状态特征 |
| end_episode | success=true/false | 回合提交确认 |
| abort_episode | episode_id | 丢弃尚未标注的回合 |

RLT helper 在断连后不自动重发消息，避免重复执行或提交。重连后先检查 status，
用原 episode_id 明确结束或 abort 尚未结束的回合，再开始新回合。

## Force 模态

train/inference 的 `openpi_rlt.yaml` 共用 `rlt.force` 结构。
`enabled: false` 不加载指内 encoder，也不要求触觉字段。
`enabled: true` 时，当前及下一状态的 `z_grip` 同时提供给 actor 和 critic，
VLA/readout 只接收原来的外部相机与腕相机输入。

`input: rgb` 使用已有的冻结 ResNet-18 编码器，需要本地 `weights`。
客户端提供四帧左右指 uint8 RGB 历史、`finger_timestamps[4,2]`、
`finger_valid_mask[4]`、`monotonic_ts`、每回合独立的 `finger_session_id`，
以及 `finger_baseline_left/right` 无接触基线。特征维度为 12288。
`input: features` 直接读取客户端的有限 `z_grip[feature_dim]`；客户端负责相同编码与时间对齐。

replay/checkpoint 校验 force 维度，模型身份包含 VLA/readout、归一化统计及 RGB encoder 权重。
不能切换同一 checkpoint 的 force 开关后直接使用。已有旧格式 replay 的模型身份计算方式
不同，需重新构建后供在线模式使用。

## 从 YAML 离线训练 actor–critic

`config/train/openpi_rlt.yaml` 默认 `rlt.stage: sft`，在演示数据上联合训练 VLA 和 readout。
force 对 sft 训练不起作用。设置 `rlt.stage: actor_critic` 后，改为从
`rlt.replay.directory` 采样训练小网络，force 维度必须与 replay 一致，
不加载或训练 VLA；总更新次数使用 `num_train_steps`，batch 使用 `rlt.learner.batch_size`。

```bash
uv run scripts/train.py --config config/train/openpi_rlt.yaml --rlt.stage actor_critic
```

`rlt.actor_checkpoint` 用于离线续训，或者使用 `--resume` 恢复
`rlt.learner.checkpoint_path`。此阶段保留原来的命令行离线训练入口。

接触实验使用第三个显式选择的 `rlt.stage: force`，通过 `rlt.force.enabled`
选择真实指内特征或恒零对照，通过 `rlt.force_stage.policy: vla` 部署冻结 VLA。
这条路径读取 `rlt.force_stage.replay_path` 的 JSONL，不使用普通 RLT 的 Pickle replay；
需要七关节参考及匹配的关节 VLA/readout。配置和统一入口见
[Force RLT](force_rlt_pytorch.md)。开启 `force.enabled` 本身不会自动切换训练 stage。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest -q \
  src/openpi/tests/serving/online_rlt_test.py src/openpi/tests/rlt
```
