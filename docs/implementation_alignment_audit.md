# OpenPI / RLinf / openpi-RLT 实现对照

审计日期：2026-10-09。以下记录修复前的发现；后续修复与验证见 [PyTorch 对齐说明](pytorch_alignment.md)。

2026-10-27 补充：[RLT 数采、关键阶段与人工接管对照](rlt_collection_alignment.md) 记录当前数据流程与参考的差异，尤其是非关键段过滤、warmup 计数、人工接管及离线数据复用。下文仍保留 2026-10-09 的历史审计内容。

修复前结论：普通 PyTorch VLA 主干与 RLinf 的核心计算一致，但加载与转换链路存在问题；当前 RL-token 和第二阶段 RLT 只是基于参考实现的改写，尚未达到与 openpi-RLT 等价的要求。该次审计尚未修改模型或训练行为。

对照的是本机工作树，而不是网上版本：

| 仓库 | 路径 | HEAD |
| --- | --- | --- |
| 当前仓库 | `/data/codes/grade_two/my_projects/robot/openpi-force` | `167035bbc0c6af30f30b5dfcafec91a36e7282d3` |
| RLinf | `/data/codes/grade_two/github/robot/rlinf` | `61ba34e640035f2e4ac4ef3c6078f02d43a00c9a` |
| openpi-RLT | `/data/codes/grade_two/github/robot/openpi-RLT` | `c1e40ac360185778c98cf20da2820e22d2d415e7` |

当前仓库有大量暂存修改，结论覆盖这些修改；两个参考仓库的 `git status --short` 均为空。当前 vendored RLinf 的 README 指向的 commit 恰好就是本机 RLinf HEAD。

## 1. PyTorch VLA 与 RLinf

当前训练入口 `scripts/train.py → training.runner → training.vla` 使用 `models_pytorch/pi0_pytorch.py`。实际 backbone 是 `models_pytorch/rlinf/`；旁边的 `gemma_pytorch.py` 和 `transformers_replace/` 属于另一套 HF 实现，不能根据那些文件判断当前主路径。

| 项目 | 对照结果 |
| --- | --- |
| `rlinf/model.py` | 与 RLinf `modules/model.py` 字节一致；Observation、图像 resize、训练增强也相同 |
| `rlinf/siglip.py` | 字节一致 |
| `rlinf/lora.py` | 字节一致 |
| `rlinf/utils.py`、`pointnet.py`、`__init__.py` | 字节一致 |
| `rlinf/gemma.py` | 除新增 headwise attention LoRA 及相应 projection 构造外，一致 |
| attention mask、时间 embedding、suffix embedding、action projection、选择性 dtype 转换 | 对应方法 AST 一致 |
| flow matching | 相同的 Beta(1.5, 1.0) 时间采样、`x_t=t*noise+(1-t)*actions`、`u_t=noise-actions` 和逐元素 MSE；当前多了 Observation 兼容转换 |
| Euler sampler / KV cache | 当前拆出共享 prefix/readout cache；固定输入下动作与 RLinf 相同 |
| gradient checkpointing | 正常 SFT forward 都关闭 activation checkpointing |

使用直接来自 RLinf 源文件的核心方法和独立加载的上游 Gemma，复制相同权重，固定 observation、actions、noise、time，检查了以下四种组合。为限制资源，用 2 层小 Gemma 和相同的 TinyVision 替代完整 SigLIP；不加载 RLinf 的分布式/runtime/task 包装。

| 模型 | dtype | loss 最大差值 | 参数梯度最大差值 | 3 步 Euler 动作最大差值 |
| --- | --- | --- | --- | --- |
| Pi0 | float32 | 0 | 0 | 0 |
| Pi0 | bfloat16 | 0 | 0 | 0 |
| Pi0.5 | float32 | 0 | 0 | 0 |
| Pi0.5 | bfloat16 | 0 | 0 | 0 |

这证明了所检查核心路径的数值一致性，不代表完整 2B 模型、GPU kernel、分布式训练、PPO/DAgger/DSRL 或 LoRA 训练均已验证一致。

**需要修复：加载 VLA 时没有恢复混合精度。**

[当前 load_pytorch](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/models/model.py:248) 只构造模型并 `safetensors.torch.load_model`。构造出的 Linear 权重默认 FP32，copy 加载不会替换参数 dtype，而 `config.dtype=bfloat16` 让 Gemma 输入转为 BF16。readout [训练入口](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/training/rlt.py:63) 和 [服务加载入口](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/policies/policy_config.py:51) 均走这条路径，只调用 `.to(device)`，没有调用 `to_bfloat16_for_selected_params`。

使用实际 `BaseModelConfig.load_pytorch` 加载小模型 safetensors，再提取 prefix，可以复现：

```text
expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float
```

普通 VLA 训练 [training/vla.py](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/training/vla.py:462) 有正确的转换；RLinf 的 [_apply_openpi_param_dtypes](/data/codes/grade_two/github/robot/rlinf/rlinf/models/embodiment/openpi/__init__.py:295) 在加载后也执行选择性转换。应统一加载行为，并保持 action heads、norm、vision stem 的 FP32 设置。现有 readout smoke 测试替换了真实 VLA，无法捕获这个问题。

**需要修复：JAX → PyTorch 转换脚本仍使用旧参数布局。**

[examples/convert_jax_model_to_pytorch.py](/data/codes/grade_two/my_projects/robot/openpi-force/examples/convert_jax_model_to_pytorch.py:56) 生成 `paligemma_with_expert.*` 的 HF 参数名，当前模型需要 `llm.*`、`img.*`。脚本在 [第 520 行](/data/codes/grade_two/my_projects/robot/openpi-force/examples/convert_jax_model_to_pytorch.py:520) 使用 `strict=False` 且忽略返回的 missing/unexpected keys，只有布局未变的动作投影能匹配，backbone 权重会保留随机初始化并被保存。小模型参数加载实验确认旧 embedding key 为 unexpected，全部 42 个小模型 `llm/img` 参数均 missing。

脚本还用整个模型 `.to(bfloat16)`，没有保持 RLinf 的 FP32 参数区域。应更新为 RLinf 参数映射，检查所有 backbone 参数已加载，再执行选择性 dtype 转换。参考 RLinf 的 [jax_to_openpi.py](/data/codes/grade_two/github/robot/rlinf/rlinf/utils/ckpt_convertor/openpi/jax_to_openpi.py:81)。这里没有运行完整 JAX checkpoint 转换，判断来自参数映射和实际 `load_state_dict` 行为。

**LoRA 属于扩展，训练方式尚不等价。**

当前默认 YAML 使用 `gemma_2b_lora / gemma_300m_lora`，本地新增 [HeadwiseLoRALinear](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/models_pytorch/rlinf/gemma.py:207)，按 head 存储低秩矩阵；上游 native attention LoRA 分支直接 `NotImplementedError`。上游的另一条 [is_lora 路径](/data/codes/grade_two/github/robot/rlinf/rlinf/models/__init__.py:375) 使用 PEFT，可配置 rank 和 target scope。它与本地 headwise adapter 的参数形状、初始化及冻结范围不同。本地 attention A/B 都随机初始化，基础权重加载后初始 adapter 更新非零；PEFT gaussian 初始化的 B 为零。因此不能将上面的无 LoRA 数值一致结论推广到默认 LoRA 训练配置。

## 2. RL-token 与 openpi-RLT

两边相同的部分是：使用 post-transformer image prefix，去掉语言输出 token，编码为少量 RL tokens，通过 decoder 重建 embedding，用 MSE 训练；readout loss 阻断对 VLA 的梯度。但具体网络不同。

主要对照文件：[当前 rl_token.py](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/models_pytorch/rl_token.py:39)、[参考 rl_token.py](/data/codes/grade_two/github/robot/openpi-RLT/src/openpi/models/rl_token.py:23)、[参考 CrossAttentionLayer](/data/codes/grade_two/github/robot/openpi-RLT/src/openpi/models/utils/fsq_tokenizer.py:269)。

| 差异 | 当前 PyTorch | openpi-RLT JAX | 影响 |
| --- | --- | --- | --- |
| Encoder query | 随机初始化可训练参数 | sinusoidal 初始化可训练参数 | 初始计算不同 |
| Encoder memory 位置编码 | 每次生成固定 sinusoidal | `self.param(y_pos_enc)`，训练中更新 | 表达能力及梯度不同 |
| Decoder query / memory 位置编码 | 固定 sinusoidal | 都是 sinusoidal 初始化的可训练参数 | 不是同一个 decoder |
| sin/cos 排列 | sin/cos 交错 | 前半 sin、后半 cos | 未映射权重时初始 embedding 不同 |
| Cross-attention memory | 额外经过 `kv_norm` | memory 直接送入 attention，仅 query 做 norm | 实际 attention 不同 |
| MLP | Linear → GELU → Linear | Dense → Dropout → GeGLU（含额外 Dense 和门控）→ Dense | 网络结构、参数量不同 |
| LayerNorm | PyTorch 默认 eps=1e-5 | Flax 默认 eps=1e-6 | 数值差异 |
| input/output projection | 总是有 Linear | input_dim=embed_dim 时不投影 | 相同宽度下多了两层可训练映射 |
| 配置 | num_heads 可调，缺少 dropout_rate；训练 YAML 未暴露 mlp_ratio | RLTokenConfig 含 dropout_rate、mlp_ratio | 配置能力不同 |

当前 self-attention → cross-attention → MLP 的大结构与参考相似，但这些差异足以认定其不是等价移植。仅把 embed_dim 改为 2048 不能解决结构问题。

**第一阶段训练也不同。**

| 项目 | 当前 training.rlt | openpi-RLT scripts/train_rlt.py |
| --- | --- | --- |
| VLA 更新 | 永远冻结，eval + no_grad | alpha=0 冻结；alpha>0 联合训练，loss=`Lro + alpha*Lvla`；Lro 始终 stop-gradient |
| 图像增强 | `extract_prefix_embeddings` 固定 train=False | 训练调用 train=True，有 crop/rotate/color jitter |
| readout attention/loss mask | 使用有效 image mask | 实际训练将 `None` 传入 RLT module，虽然模块本身支持 mask |
| LR | 固定 rlt.learning_rate；默认 YAML 3e-4 | TrainConfig lr_schedule，默认 warmup 1000，peak 2.5e-5，cosine decay |
| AdamW | YAML beta2=0.999、weight_decay=0.01 | 默认 beta2=0.95、weight_decay=1e-10 |
| EMA | 无 | TrainConfig 默认 ema_decay=0.99，并执行 EMA 更新 |
| 恢复训练 | readout resume 明确禁止；只保存权重/manifest/config | 恢复 train_state、optimizer 等状态 |
| 训练并行 | 只允许单进程 | JAX mesh/FSDP |
| 默认模型实例 | Pi0、16 步、embed_dim=512、Franka | 具体 rlt_pi05/agilex 配置为 Pi0.5、50 步、embed_dim=2048 |

依据：[当前 train](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/training/rlt.py:63)、[参考 loss 分支](/data/codes/grade_two/github/robot/openpi-RLT/scripts/train_rlt.py:151)、[参考训练及恢复](/data/codes/grade_two/github/robot/openpi-RLT/scripts/train_rlt.py:285)、[参考 optimizer](/data/codes/grade_two/github/robot/openpi-RLT/src/openpi/training/optimizer.py:74)、[参考配置](/data/codes/grade_two/github/robot/openpi-RLT/src/openpi/training/config.py:1136)。

mask 差异在 [迁移计划](/data/codes/grade_two/my_projects/robot/openpi-force/FORCE_RLTOKEN_MIGRATION_PLAN.md:35) 中明确要求过，是有意调整；有效相机全部存在时它可能没有效果，缺失相机被 mask 时会改变训练目标。模型、相机、horizon 和动作维度的机器人适配同样可以保留，但应与“算法严格一致”的要求区分并写明。用户现在要求等价，需要明确兼容模式，不能把这些调整默认为已对齐。

另一个易混淆点：**RLinf 自己的 RLT 模块也不是 openpi-RLT 的 readout。** RLinf [rlt_token_transformer.py](/data/codes/grade_two/github/robot/rlinf/rlinf/models/embodiment/modules/rlt_token_transformer.py:103) 使用 prefix+RL token 联合 self-attention encoder 和带 shifted targets 的 causal teacher-forcing decoder。按本次要求，应以 RLinf 对齐 VLA，以 openpi-RLT 对齐 readout，不能直接将 RLinf 的 RLT 模块作为替代。

## 3. 第二阶段 actor / critic / replay

主要对照：[当前 actor](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/actor.py:27)、[当前 critic](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/critic.py:20)、[当前 learner](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/learner.py:107)、[参考 networks](/data/codes/grade_two/github/robot/openpi-RLT/rlt_online_rl/src/rlt_online_rl/networks.py:57)、[参考 trainer](/data/codes/grade_two/github/robot/openpi-RLT/rlt_online_rl/src/rlt_online_rl/trainer.py:175)。

| 项目 | 当前普通 RLT | openpi-RLT |
| --- | --- | --- |
| actor 输入编码 | 原始 z/proprio/ref 拼接后直接 MLP | 独立 z→256、proprio→64、ref→256 投影，norm/tanh 后拼接 |
| actor 输出 | H 步 `ref + offset*tanh(residual)` | E 步直接预测均值 mu |
| reference dropout | 只丢弃 conditioning；输出始终加原始 ref | 丢弃 actor 收到的 ref 输入，无额外恒定 residual anchor |
| critic | 原始输入拼接，动作 mask 并加入 mask 特征 | 独立 z/proprio/action 投影，之后 MLP；不输入 mask |
| LayerNorm / 初始化 | 可训练 affine LN，eps=1e-5；Linear 默认初始化 | 无可训练 affine LN，eps=1e-6；Glorot uniform / zero bias |
| actor Q 项 | `-q_weight * mean(min(Q1,Q2))` | `-q_weight * mean(Q1)` |
| 平滑项 | 普通 learner 没有 delta_weight | 支持物理空间前六维步间差分正则；AgileX task 配置 delta_weight=10 |
| optimizer | AdamW，未指定 weight_decay，因此默认为 0.01 | optax.adam，无 weight decay |
| actor 更新 | 每 2 次 critic update（默认） | 相同 |
| target 更新 | 每次 critic update 后更新 actor/critic targets | 只在 actor update 分支更新两个 targets |
| BC 人工目标 | HUMAN 步使用 executed action，其余用参考；padding 排除 | HUMAN 或 MIXED 步使用 executed action；BC 普通均值 |
| actor BC 范围 | 只监督前 E 步，actor 仍预测 H 步 | actor 本身输出 E 步，监督完整输出 |

Gaussian 固定方差采样、twin critic MSE、target bootstrap 取 min(Q1,Q2)、reference dropout 的总体思路一致。完整 E 步窗口的 TD 公式一致：`sum(gamma^i*r_i) + (1-done)*gamma^E*min(next_Q)`。当前 partial window 必须终止，所以有效步数指数不同在那些样本上不会产生额外 bootstrap；实际差异主要在网络输入、padding 的 critic/BC 处理和样本构建。

replay 的主要差异：

- 当前显式记录 executed_mask，保留短回合，并将末端窗口对齐终止奖励；参考通用 builder 默认 `allow_partial=True`，按 stride 继续创建部分窗口，另有 terminal-aligned helper。两边窗口数量/分布不相同。
- 当前保存当前状态的完整 H 步 VLA 参考，参考通用 step builder 将逐步 `ref_action` 拼成 E 步 chunk；具体 online 路径还会保存 chunk 起点缓存。两边 replay schema 不可直接互换。
- 当前只支持 uniform ring-buffer sampling；参考还支持 stratified sampling，默认同样是 uniform。因此默认 sampling 方法本身不是差异，启用 stratified 时才是。
- 当前 action adapter 对多臂 EE 的 RPY 差分做角度环绕；参考 delta_chunk 直接减前六维，并保留全零 padding。一般非零、单臂、无角度跨界动作的分位数变换公式相同，边界及 padding 行为不同。
- 当前示例 actor/critic 使用物理绝对动作、30 Hz；AgileX task 示例使用带统计的 delta_chunk、20 Hz。这是具体任务配置差异，不能直接复制另一台机器人的数值；动作单位不同会改变 std、offset、BC/Q 权重的实际含义。

依据：[当前 episode builder](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/episode.py:115)、[参考 builder](/data/codes/grade_two/github/robot/openpi-RLT/rlt_online_rl/src/rlt_online_rl/replay.py:328)、[当前 adapter](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/action_representation.py:49)、[参考 adapter](/data/codes/grade_two/github/robot/openpi-RLT/rlt_online_rl/src/rlt_online_rl/action_representation.py:106)、[AgileX task config](/data/codes/grade_two/github/robot/openpi-RLT/rlt_online_rl/configs/tasks/agilex_ethernet/online_rl.yaml:1)。

`rlt/force_rlt/` 是另一套接触扩展，也不能视为等价第二阶段：增加 z_grip/q/ref 的输入，使用 residual actor；[force learner](/data/codes/grade_two/my_projects/robot/openpi-force/src/openpi/rlt/force_rlt/learner.py:101) 拒绝 HUMAN/intervention，每步更新 actor 和 targets，BC 监督完整 H 步，smooth 在全部动作维度上计算。普通 RLT 的可选 force features 与这个专用 force learner 也不是同一条训练路径。触觉输入和机器人适配可以是合理扩展，但其优化行为应单独命名、记录。

## 4. 验证与建议顺序

已验证：

- 模型与 RL-token、普通 learner、replay 测试：16 passed。
- readout 训练/服务/WebSocket smoke：4 passed。沙箱最初禁止本机端口绑定；允许本机测试后通过。该 smoke 使用 FakeVLA，不能证明真实加载路径无问题。
- 独立小模型 RLinf 对照：4 种组合下 loss、梯度、采样差值均为 0。
- 实际 safetensors/load_pytorch 小模型加载：复现 bfloat16 dtype mismatch。
- 旧 converter key 加载实验：确认旧主干参数名不被当前模型接收。

命令（Hugging Face 缓存重定向到可写 `/tmp`）：

```bash
HF_HOME=/tmp/openpi-audit-hf HF_DATASETS_CACHE=/tmp/openpi-audit-hf/datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python -m pytest -q src/openpi/tests/models_pytorch \
  src/openpi/tests/rlt/learner_test.py src/openpi/tests/rlt/replay_test.py

HF_HOME=/tmp/openpi-audit-hf HF_DATASETS_CACHE=/tmp/openpi-audit-hf/datasets \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
.venv/bin/python -m pytest -q src/openpi/tests/serving/rlt_pytorch_smoke_test.py
```

临时主干对照脚本保存在 `/tmp/openpi_parity_audit.py`。本机没有可用 CUDA；没有跑完整 checkpoint/GPU 训练或真机 rollout。现有测试通过证明当前设计内部可运行，不能替代跨实现的参数映射及数值一致性验证。

建议修复顺序：先统一 VLA 加载的选择性 dtype 转换，并修复 JAX converter 的映射和完整性检查；再按 openpi-RLT 移植 readout 的可训练位置参数、GeGLU、norm/projection 行为；然后补齐 alpha 联合训练、训练增强、可配置 mask、scheduler/EMA/resume；最后对齐普通 actor/critic 投影、直接输出、Q1 loss、Adam、target 更新频率和 delta 正则。机器人维度、有效动作 mask、触觉输入、安全残差等适配应作为明确的扩展保留，并为严格参考模式提供独立配置。

验收应分别覆盖：相同 VLA 权重/输入的 loss 和动作一致；JAX→Torch 映射后 readout token/reconstruction/loss 一致；相同 replay 的 actor/critic forward、loss、梯度和一步参数更新一致。仅检查输出形状、loss 下降或 smoke 成功不足以证明实现等价。
