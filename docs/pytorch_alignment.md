# PyTorch、LoRA 与 RLT 训练对齐

当前实现对照本机 RLinf `61ba34e6` 和 openpi-RLT `c1e40ac3`。普通 VLA 使用 RLinf 的 PyTorch Pi0/Pi0.5 主干、flow matching 和 Euler sampler；第一阶段的 readout 使用 openpi-RLT 的 cross-attention encoder/decoder。

openpi-RLT 的对应源码快照保存在 [third_party/openpi-RLT](../third_party/openpi-RLT/README.md)，版本和来源见[快照记录](../third_party/openpi-RLT.UPSTREAM.md)。

## VLA 精度与 LoRA

训练和推理加载都使用 `pytorch_training_precision`，默认 `bfloat16`。Gemma/SigLIP 主干采用 BF16，动作与时间投影、Gemma norm、视觉 patch embedding 和位置 embedding 保持 FP32，与 RLinf 的选择性转换一致。`float32` 会同时恢复权重和计算 dtype，避免参数 dtype 与输入 dtype 不一致。

FP32 保留参数直接维持 FP32，不先经 BF16 再转回，避免完整 checkpoint 恢复时截断视觉或 norm 权重的精度。

`model.is_lora` 显式开启或关闭 LoRA；未设置时，`*_lora` 模型名称启用 LoRA。`model.lora_mode` 选择训练范围：

| 模式 | Gemma 语言模型与动作专家 | 视觉编码器与外部动作/状态/时间投影 |
| --- | --- | --- |
| `lora_only`（默认，所有原始权重冻结） | PEFT 线性层 LoRA；基础参数冻结 | 线性层 PEFT LoRA；基础参数冻结 |
| `openpi` | PEFT 线性层 LoRA；基础参数冻结 | 无 adapter，原始参数全部可训练 |

`openpi` 对应官方 JAX 示例只冻结 `llm` 非 LoRA 权重的训练范围；Gemma 的 adapter 仍使用 PEFT，不复刻 JAX 原生 headwise/FFN LoRA 的参数结构、初始化或各 expert 的 rank，也不改变现有 BF16/FP32 策略。因此两种框架的训练轨迹不保证一致。`model.lora_rank` 默认 32，控制当前两种模式的 PEFT rank；alpha 等于 rank，dropout 为 0，A 使用 Gaussian 初始化，B 初始化为零。RLinf 的原生 attention LoRA 分支没有实现，因此不再构造之前添加的 headwise adapter。`is_lora: false` 时不注入 adapter，保持全参训练。

基础 checkpoint 先严格加载，再注入零更新 PEFT；完整 PEFT checkpoint 先按配置模式注入适配器，再严格恢复。两种模式的 checkpoint 参数结构不同，加载和续训须保持模式与 rank 一致。切换模式时使用 dense 基础权重和新的实验目录。旧配置没有 `lora_mode` 时按 `lora_only` 加载，保持原有的 all-linear LoRA 训练范围。旧 headwise PyTorch checkpoint 会被明确拒绝，不会被当作新 PEFT checkpoint 使用。

## JAX checkpoint 转换

旧脚本生成 `paligemma_with_expert.*` 参数，而当前主干要求 `llm.*` 和 `img.*`。旧版 `strict=False` 会漏载主干并保存随机初始化的权重。新版直接输出 RLinf 布局，在写入前检查完整参数集合及形状，并应用相同精度规则。

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /path/to/jax_checkpoint \
  --config-name config/train/openpi.yaml \
  --output-path /path/to/new_torch_checkpoint \
  --precision bfloat16
```

`config/train/openpi.yaml` 在 `pytorch_weight_path` 下没有 `model.safetensors` 时，用从 RLinf 复制的 `src/openpi/models_pytorch/jax_to_openpi.py` 把 `jax_weight_path` 转一次，写成 fp32，然后训练再按精度策略转 bf16。文件已经存在则不再转换。默认来源是 `gs://openpi-assets/checkpoints/pi0_base`；`model.pi05: true` 时改为 `pi05_base`。官方权重是 dense 主干，训练加载后再挂零更新 PEFT。

上面的手动命令会按训练 YAML 做严格校验，并把原生 JAX LoRA 合并进 dense 权重。优化器状态不会迁移。转换会复制 normalization assets。已经由旧脚本导出的文件无法恢复被丢弃的主干，必须从原始 JAX checkpoint 重新转换。

## 第一阶段代码默认值

这些默认值属于 RL-token 第一阶段，普通 `mode: vla` 不使用它们；其中的增强、调度和优化器也用于普通 VLA。第一阶段默认冻结 VLA，只训练 readout；`rlt.alpha > 0` 时同时更新 VLA 的可训练参数：

| 项目 | 默认行为 |
| --- | --- |
| loss | `L_readout + rlt.alpha * L_vla`，默认 `alpha=0`；readout 输入 stop-gradient |
| 图像 | 训练时启用 RLinf 图像增强；只重建 post-transformer 图像 token |
| mask | VLA 保留有效相机/文本 attention mask；readout 和重建 loss 不使用 mask，与参考训练调用 `None` 一致 |
| 学习率 | warmup 1000 步，peak `2.5e-5`，cosine 到第 30000 步的 `2.5e-6` |
| 优化器 | AdamW，betas `(0.9, 0.95)`，eps `1e-8`，weight decay `1e-10`，全局梯度裁剪 1 |
| EMA | decay `0.99`，同时维护 VLA/readout |
| 恢复 | `train.resume: true` 恢复模型、优化器、EMA、数据位置、各 rank 随机状态和 step |

`rlt.alpha` 控制联合 loss。`training/rlt.py` 的 `ReadoutDefaults` 保留增强、mask 和 EMA 默认值。`rlt.embed_dim` 默认 2048，与 Gemma 2B 宽度一致，不增加输入/输出投影；设为不同宽度时增加两层 Linear。已有通用学习率/优化器配置仍可覆盖；readout 的 `rlt.learning_rate` 为该阶段 peak learning rate。

每个数字 checkpoint 同时保存匹配的 EMA `model.safetensors`、`readout.safetensors`、assets 和 `training_state.pt`。推理传入 readout 根目录时选择最新完整数字 checkpoint，并使用其中配套的 VLA，避免根目录导出过程中混用两步的权重。旧 readout 由于 query、位置参数、GeGLU、norm 和投影结构变化，需要重新训练。

## 多卡与梯度累积

普通 VLA 和 `rlt.stage: sft` 使用 PyTorch DDP。`scripts/train.py` 按 `train.num_gpus` 自动启动进程；每卡保留完整模型，读取不同数据，同步梯度。这里没有实现 RLinf 的 FSDP 参数/优化器分片；DDP 不能解决单卡放不下完整模型的问题。RLinf 的 `NO_SHARD` 配置同样复制模型，分片能力取决于它的具体配置。

普通 VLA YAML 默认 `train.micro_batch_size: null`，不做梯度累积，每卡一次处理 `train.batch_size / train.num_gpus` 个样本；单卡 batch 5 就是整批 5 条一起计算。设为整数后，累积次数为 `train.batch_size / (train.micro_batch_size * train.num_gpus)`，必须整除。梯度累积也可用于单卡：batch 5、micro batch 1、单卡时累积五次；batch 8、micro batch 1、两卡时累积四次。RLT YAML 仍显式使用 micro batch 1。默认 batch 5 不能直接配两卡或四卡。前向和反向都放在 DDP `no_sync` 内，最后一次 micro batch 才同步；数据 sampler 每个 epoch 重新洗牌。RLT 的冻结与联合路径通过同一个 DDP 模块计算 loss。

RLT checkpoint 保存每个 rank 的 CPU、CUDA 和 NumPy RNG 状态，恢复时各自加载。恢复要求保持卡数、batch 和其他训练设置一致。旧单进程 checkpoint 的随机状态仍可恢复；没有各 rank 随机状态的旧多进程 checkpoint 会明确报错。

## 第二阶段 YAML 配置

`config/train/openpi_rlt.yaml` 和 `config/server/openpi_rlt.yaml` 提供相同的小网络设置：

| 字段 | 默认值与用途 |
| --- | --- |
| `rlt.actor.input_projection` / `rlt.critic.input_projection` | `true`：z/proprio/action 分别投影到 256/64/256，使用非 affine LayerNorm、eps `1e-6` |
| `rlt.actor.output_mode` | `direct`：直接预测 E 步，H-E 尾部沿用 VLA reference；`residual` 可使用残差输出 |
| `rlt.learner.actor_q_reduction` | `q1`：actor loss 使用 Q1；`min` 可改为两个 Q 的最小值；TD target 始终使用 min Q |
| `rlt.learner.optimizer` / `weight_decay` | `adam` / `0.0`；也支持 `adamw` |
| `rlt.learner.target_update_period` | `null`：随 actor 更新；可指定 critic 更新计数周期 |
| `rlt.learner.delta_weight` / `smooth_action_dims` | `0.0` / `6`：在物理空间比较预测和 BC target 的相邻步增量，忽略 padding；多臂动作应按实际布局调整 |

全有效 E 步情况下，这些网络和损失对应参考普通 RLT。当前还保留短 chunk mask、多臂输入、可选触觉和 H 步服务协议；这些是本项目扩展。旧 actor/critic checkpoint 结构不兼容，新格式为 `openpi-rlt-learner-v2`。独立 `force_rlt` B/C 实验网络保持自己的实现。

## 数采与在线流程的对齐范围

网络和损失的数值对照不代表数采流程等价。2026-10-27 核对发现：参考 ROS 链路在 warmup 和 online 都排除非关键阶段的 replay；当前普通在线服务的 `critical_phase_only` 只限制 actor 接管，非关键段反馈仍入库并计入 warmup。参考支持同回合人工接管再交回策略，当前在线 `HUMAN` 反馈后要求结束回合；当前离线导入支持混合来源回合，但没有关键阶段裁剪入口。

完整示教、关键阶段切换、600 条统计、`c` / `t` 的操作语义、离线数据复用和待补功能见 [RLT 数采、关键阶段与人工接管对照](rlt_collection_alignment.md)。该文区分当前行为与待实现要求。

## 验证范围

使用真实小 Gemma/SigLIP 检查 Pi0/Pi0.5 × FP32/BF16 的完整转换、严格加载和 PEFT checkpoint 恢复；联合训练检查连续两步与一步后恢复的 VLA/readout/EMA 和随机状态一致。readout 的 token、重建、loss 和梯度与参考 Flax block 数值对照通过；普通 actor/critic 固定参数与本机 openpi-RLT 的误差小于 `1e-6`。这些检查验证了代码路径，尚未验证完整 2B GPU 训练的任务成功率。
