# 两个训练入口与参考实现核查（2026-10-10）

结论：相关自动测试通过，但当前两个 YAML 不能直接启动真实数据训练。Torch VLA 的核心网络、loss、梯度和采样与本机 RLinf 一致；RL-token 的主要算法流程已实现，仍有训练范围、采样和运行方式的差异。CPU 验证不代表完整预训练模型的 GPU 训练或机器人任务效果已经验证。

对照本机 `openpi-RLT` HEAD `c1e40ac` 与 `rlinf` HEAD `61ba34e6`；被核查的参考文件没有工作树修改。

## 1. 测试结果与实际启动

最终：**110 passed，1 deselected**，耗时 93.84 秒。范围：`tests/training`、`tests/models_pytorch`、`tests/rlt`、`tests/serving`。排除的是 `manual` 真实数据集测试。

覆盖普通 VLA 的 CPU 训练/保存/恢复，Pi0/Pi0.5 的 FP32/BF16、PEFT、转换和严格加载，RLT 联合训练/EMA/恢复，LeRobot v3 数据读取/归一化，replay/actor/critic，以及 YAML 调度和真实本地 WebSocket 往返。

第一轮有一处旧测试断言失败：`jax.tree.leaves(batch)` 将新的 Torch `Observation` 当作叶子，访问 `.shape` 报错。已在 `src/openpi/tests/training/data_loader_test.py` 改为遍历 `(observation.to_dict(), actions)`，保留每个 tensor 的 batch 维度检查。没有修改训练实现或训练 YAML。其余环境失败分别由可写 `/tmp` HF 缓存和允许临时本地端口监听解决。

复现命令（使用已有 `.venv`，不同步/下载依赖）：

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
UV_CACHE_DIR=/tmp/openpi-audit-uv \
HF_HOME=/tmp/openpi-audit-hf HF_DATASETS_CACHE=/tmp/openpi-audit-hf/datasets \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --offline --no-sync python -m pytest -q -m 'not manual' \
  src/openpi/tests/training src/openpi/tests/models_pytorch \
  src/openpi/tests/rlt src/openpi/tests/serving
```

这里的 WebSocket 测试需要能监听 `127.0.0.1`。日志：`/tmp/openpi-training-audit.log`；JUnit：`/tmp/openpi-training-audit.xml`。

实际分别运行两个训练入口，加 `--offline --no-sync` 并隐藏 CUDA，均在加载完整模型之前报错：

```text
FileNotFoundError: LeRobot dataset metadata not found:
/data/codes/grade_two/my_projects/robot/data_collect/data/dataset/franka_lerobot_test_100hz/meta/info.json
```

| 项目 | 当前状态及处理 |
| --- | --- |
| 数据集 | 两份 YAML 的默认 repo_id 对应目录没有 metadata。配置实际 repo_id，或通过 `OPENPI_DATASET_ROOT` 指向符合字段、30 Hz 和动作时序要求的采集数据。 |
| 普通 VLA 归一化 | 当前 `data.create(...)` 返回 `norm_stats=None`。需计算与该 YAML 动作表示一致的统计，或配置已有匹配的 assets。 |
| 基础 Torch 权重 | 已有 `data/checkpoints/torch/pi0_base/model.safetensors`，约 13 GB、FP32。通过 safetensors header 与 meta-device 完整模型核对：632 个 tensor，无缺失/额外 key，无 shape 不匹配，共 3,238,048,528 个参数。此检查没有把全部权重载入内存，也不能验证数值内容或任务效果。 |
| RLT 初始 VLA | `rlt.vla_checkpoint=null`。需提供匹配的 VLA checkpoint，包含 `model.safetensors` 与对应 normalization assets。基础权重目录当前没有这些数据集 assets。 |
| 串联训练 horizon | 普通 VLA 为 50，RLT 为 16。若用第一个输出训练第二个，应统一 horizon，并同步服务端配置；当前 RLT 入口没有检查初始 checkpoint 的配置是否匹配。horizon 通常不改变权重 shape，不能依赖 strict load 发现这个问题。 |

本环境 `nvidia-smi` 无法连接驱动；没有运行完整模型的 GPU 前向、反向或真实数据长训练。启动日志分别为 `/tmp/openpi-startup-audit.log`、`/tmp/openpi-rlt-startup-audit.log`。

## 2. 与 openpi-RLT 的 RL-token 流程对照

已具备的流程：VLA 提取 post-transformer 图像 prefix → stop-gradient/FP32 → cross-attention encoder 压缩为 token → decoder 重建/MSE → `L_readout + alpha * L_vla` → 配套 VLA/readout EMA checkpoint → 冻结模型提取 reference/token → replay → actor/twin critic、TD、BC 与 target 更新 → warmup/online 服务及执行反馈。

readout 的可学习 sinusoidal query/position、self/cross attention、GeGLU、LayerNorm eps、条件输入/输出投影与参考源码一致。默认 reconstruction 不使用 mask，符合参考 `train_rlt.py` 传入 `None` 的行为。第一阶段同时保存 EMA VLA/readout、训练状态、优化器和随机状态；现有测试检验连续两步与中途恢复的一致性。

直接加载参考 `src/openpi/models/rl_token.py` 对照四组小模型（投影开/关 × mask 开/关，2 层）：token、重建、MSE、encoder/decoder query 梯度均通过数值比较，最大绝对误差约 `6.68e-6`。同参数下参考 actor 输出误差 `8.34e-7`，Q1 `8.05e-7`，Q2 `1.79e-7`。actor/critic 对照采用无触觉、完整有效 E 步、直接输出模式。

仍有以下差异或缺项：

1. **联合训练更新范围不同。** 参考 `train_rlt.py` 在 alpha>0 时训练全部参数；这里默认 `*_lora`，基础参数冻结，仅更新 PEFT adapter 与 readout。数学目标一致，训练轨迹不等同。需要复现全参分支时，应显式关闭 LoRA，并使用兼容的 dense checkpoint。已训练的 PEFT checkpoint 不能只把 `is_lora` 改为 false 就加载。
2. **第一阶段只有单进程。** `training/rlt.py` 对 WORLD_SIZE!=1 明确报错；参考支持 JAX mesh/FSDP。这里没有 RLT Stage 1 的 DDP/FSDP。
3. **alpha/增强/mask/EMA 的控制入口不同。** 这里是 `ReadoutDefaults` 代码选项，默认 alpha=1；参考可通过配置选择 alpha=0 冻结 VLA。当前 YAML 无法切换这些代码默认值。readout peak LR 使用 `rlt.learning_rate`，会覆盖通用 `lr_schedule.peak_lr`。
4. **缺少可选分层 replay 采样。** 参考支持 `uniform` 和 `stratified`，后者按近期 online、warmup demo、人工干预比例取样；这里仅有均匀有放回采样，transition 也没有完整的 collection phase/intervention 标记。参考默认也是 uniform，所以不是默认算法路径的必需缺项。replay 少于 batch 时，参考会缩小 batch；这里仍按请求大小重复抽样。
5. **数据/动作/部署协议经过适配。** 这里使用 Franka EE/关节数据、短 chunk mask、H 步尾部 reference 和可选触觉。参考的 AgileX 数据、delta_chunk 动作及双机器 HTTP/ROS 服务不能直接替换进来。逐步 `MIXED` 标签在参考中视为人工动作；这里逐步标签仅允许 BASE/RL/HUMAN，MIXED 只由整个窗口推导，导入参考 replay 时需要显式转换。

参考文件：`openpi-RLT/scripts/train_rlt.py`、`src/openpi/models/rl_token.py`、`rlt_online_rl/src/rlt_online_rl/{networks,trainer,replay}.py`；本仓库对应 `training/rlt.py`、`models_pytorch/rl_token.py` 和 `rlt/`。

## 3. 与 RLinf 的 Torch VLA 对照

本仓库 `models_pytorch/rlinf/{gemma,siglip,model}.py` 与参考对应模块逐文件相同。Pi0 包装层增加 OpenPI 接口及 image-prefix 输出；flow matching、attention mask、位置、Pi0 state/time 和 Pi0.5 adaRMS 路径、Euler sampler 与选择性 BF16/FP32 保持一致。LoRA 对应 RLinf 的 PEFT `all-linear` 分支：基础参数冻结、alpha=rank、Gaussian A、零初始化 B。

扩大数值验证使用真实小 Gemma/SigLIP，直接提取参考 Pi0 的核心方法，省去依赖 RLinf 分布式系统的 runtime 初始化；没有运行 RLinf 的完整 worker。八组组合为 Pi0/Pi0.5 × FP32/BF16 × dense/LoRA，复制相同参数、固定输入/noise/time/增强随机状态。LoRA 的 B 设为非零，验证实际 adapter 更新路径。结果：增强训练 loss、参数梯度、3 步 Euler 动作、相同超参数下梯度裁剪后一次 AdamW 更新的最大差值全部为 **0**。为隔离内核编译，此对照关闭融合 kernel 的 torch.compile；常规测试覆盖默认路径。

因此可以说 **Torch VLA 核心计算已对齐**，但整个训练系统还不是 RLinf 的等价实现：

- 本仓库普通 VLA 使用 DDP 单 batch 更新；RLinf worker 支持 FSDP、micro-batch 梯度累积和全局 batch。需保持有效 batch、模型、数据变换和更新次数一致，才有比较训练轨迹的基础。
- 这里使用 OpenPI warmup/cosine，RLinf 要显式选择 `openpi_cosine`/`ref_warmup_cosine` 并匹配 warmup、总步数、min_lr。某些 RLinf 示例只写 warmup_style，其 scheduler 默认实际是 constant，不能据字段名称假定两者一致。
- 普通 VLA 保存模型/优化器/step，但没有保存训练 RNG 和数据位置，恢复仅保证可续训。RLinf 在 StatefulDataLoader 分支支持恢复 data/rng；第一阶段 RLT 已有自身的恢复实现。
- 本仓库数据仍由本地 Franka/OpenPI transform 处理，包括 EE delta、30/100 Hz 对齐和归一化；需要统一数据与统计才能比较实际训练结果。这里默认 SFT 对完整 padded horizon/action_dim 求平均，与 RLinf 普通 SFT 默认一致；没有提供其 `use_action_chunk_loss=true` 的裁切选项。

## 4. 额外发现与处理优先级

普通 VLA 的最终 checkpoint 判断存在边界问题：调用 `save_checkpoint` 前已 `global_step += 1`，内部却判断 `global_step == num_train_steps - 1`。实测 `num_train_steps=3, save_interval=1000` 时传入完成步数 3，不保存末步 checkpoint。当前 30000/1000 因末步整除间隔，仍会保存；一两步 smoke run 或其它非整除步数会受影响。现有训练测试将 save_interval 设为 1，未覆盖这一边界。本次保留训练实现，只报告问题。

建议依次处理：实际数据/统计与 RLT 初始 checkpoint → 统一串联 horizon 并校验初始 checkpoint 配置 → VLA 最终保存条件 → 普通 VLA 精确恢复 → 按实验需求增加 Stage 1 分布式和分层 replay。要复现参考全参联合训练，先明确关闭 LoRA；当前默认可作为 LoRA 联合训练方案。

额外对照脚本与结果位于 `/tmp/openpi_training_parity_20261010.py`、`/tmp/openpi-vla-parity-results.json`、`/tmp/openpi-rlt-parity-20261010.py`、`/tmp/openpi-rlt-parity-results.json`。这些使用本机参考路径；`/tmp` 文件可能随系统清理消失。
