# MHLA 接入 pi0 的改造计划

## 目标

本文档的目标是把当前 `MHLA` 仓库中的注意力思路，迁移到 `openpi` / `pi0` 的联合 transformer 主干中，优先解决以下问题：

1. 降低长序列训练的显存与时延开销。
2. 在尽量不牺牲动作建模能力的前提下，替换 `pi0` 中最昂贵的 attention 计算。
3. 先做最小可运行版本，再逐步支持 hybrid、cache、推理优化。


## 结论先行

对 `pi0` 来说，`MHLA` 最适合替换的不是单独的视觉塔，也不是单独的动作头，而是 **PaliGemma language model + Gemma action expert 的联合 self-attention 计算路径**。

原因是 `pi0` 的核心 forward 并不是让两路模型完全独立运行，而是在每一层里分别取两路的 `q / k / v`，拼接后做一次联合 attention，再拆回各自分支。因此，如果只修改 VLM 分支，收益会比较局部；真正的主要开销和信息交互点都在联合 attention 这一步。


## 当前结构理解

### 1. Prefix 路径

`pi0` 的 prefix 由图像 token 和语言 token 组成，图像通过 `SigLIP` 编码，语言通过 `PaliGemma` 的 token embedding 编码。prefix 内部是双向可见的。

关键代码位置：

- `openpi/src/openpi/models_pytorch/pi0_pytorch.py` 中的 `embed_prefix`


### 2. Suffix 路径

`pi0` 的 suffix 由 state、noisy actions、timestep 组成。训练时 suffix 会和 prefix 一起参与联合 attention，其中 suffix 具有 prefix-LM / causal 风格的 mask 约束。

关键代码位置：

- `openpi/src/openpi/models_pytorch/pi0_pytorch.py` 中的 `embed_suffix`


### 3. 联合 attention 路径

`pi0` 真正关键的一步在 `PaliGemmaWithExpertModel.forward()` 的联合层计算逻辑中：

1. 分别从 `paligemma.language_model.layers[layer_idx]` 和 `gemma_expert.model.layers[layer_idx]` 取 `q_proj / k_proj / v_proj`
2. 在 token 维拼接两路 `query / key / value`
3. 应用 rotary embedding
4. 调用 Gemma 的 eager attention 核心计算
5. 再按 prefix / suffix 的长度切回两路，分别过 `o_proj`、MLP、残差

这一步是 `MHLA` 最值得接入的位置。


## 为什么不能直接把 MHLA NLP 实现拿来即插即用

当前 `mhla_nlp` 里的实现虽然已经是语言建模版本，但和 `pi0` 存在几个关键差异：

### 1. `pi0` 使用的是任意 4D attention mask

`pi0` 训练时会构造完整的 4D mask，用来表达：

- prefix 内双向可见
- suffix 对 prefix 可见
- suffix 内部是 causal / block-causal

而当前 `mhla_nlp/fla/layers/mhla.py` 里的实现只接受 2D padding mask，不支持通用 3D / 4D attention mask。


### 2. `pi0` 的联合前向是“手工拼 QKV”

`openpi` 当前不是简单调用 HF 模块的 `self_attn.forward()`，而是在 `gemma_pytorch.py` 里手工取投影后的 `q / k / v` 并统一计算 attention。这意味着：

- 不能只替换 HuggingFace Gemma 的 attention 类
- 必须在 `compute_layer_complete()` 这个联合路径里接入 MHLA


### 3. 推理阶段存在 prefix KV cache

`pi0` 推理会先对 prefix 预计算 KV cache，然后在 denoising 时只对 suffix 做后续步骤。如果要让 `MHLA` 真正用于在线推理，需要支持：

- prefix 预填充
- suffix 递推 / cache 复用
- 与当前 position_ids / mask 逻辑对齐


## 推荐的改造路线

建议分三阶段推进，不建议一开始就做完整替换。


## 阶段 A：训练优先的最小可运行版本

### 目标

先让 `pi0` 在训练路径中跑通 `MHLA`，暂时不追求推理 cache 的完整支持。


### 改动范围

#### 1. 新增一个联合 attention 实现文件

建议在 `openpi` 中新增：

- `openpi/src/openpi/models_pytorch/mhla_joint_attention.py`

职责：

- 接收已经投影好的 `query_states / key_states / value_states`
- 兼容 prefix + suffix 的联合序列布局
- 支持 rotary 后的张量输入
- 输出与 Gemma eager attention 相同形状的 `att_output`


#### 2. 在联合层前向中增加可切换分支

修改：

- `openpi/src/openpi/models_pytorch/gemma_pytorch.py`

做法：

- 保留现有 eager attention 路径作为 baseline
- 在 `compute_layer_complete()` 中增加分支：`if use_mhla: ... else: eager_attention_forward(...)`
- 第一版只在训练路径开启 `use_mhla`


#### 3. 在 `pi0` 配置里增加开关

建议在 `openpi` 配置中增加以下字段：

- `joint_attn_impl: eager | mhla`
- `joint_attn_layers: all | list[int]`
- `mhla_block_size`
- `mhla_mode: hybrid | full`


### 阶段 A 的实现原则

1. 先只做 `hybrid`，不要全层替换。
2. 优先替换后半层，而不是前半层。
3. 先保留当前 `o_proj`、MLP、残差、RoPE、mask 构造逻辑不变。
4. 只替换 attention 主计算，不动 tokenization、observation preprocessing、action head。


## 阶段 B：支持 `pi0` 的 mask 语义

### 目标

让 `MHLA` 正确支持 `pi0` 当前使用的 attention 语义，而不是退化成简单 causal LM。


### 需要解决的问题

#### 1. Prefix 双向、suffix 因果的混合 mask

最稳的做法不是硬套 `mhla_nlp` 现成接口，而是在新的 `mhla_joint_attention.py` 中显式支持以下结构：

- prefix 区域：双向可见
- suffix 到 prefix：允许可见
- suffix 内部：因果可见
- padding：正常屏蔽


#### 2. 与 `block_size` / token 分组设计对齐

`MHLA` 的关键超参是 token-level 组块大小。对 `pi0` 而言，需要特别注意：

- prefix token 和 suffix token 的分布非常不均匀
- 图像 token 通常远多于 state token
- action horizon 会影响 suffix 长度

建议初始实验：

- 只对 suffix + 后半段 prefix 做分组
- 或在联合序列上直接做统一 block，但先从较小 `block_size` 开始


## 阶段 C：推理 cache 与在线部署

### 目标

使 `MHLA` 兼容 `pi0.sample_actions()` 中的 prefix cache + suffix denoising 推理流程。


### 需要覆盖的路径

修改和验证：

- `openpi/src/openpi/models_pytorch/pi0_pytorch.py` 中的 `sample_actions`
- `openpi/src/openpi/models_pytorch/pi0_pytorch.py` 中的 `denoise_step`


### 关键问题

1. `MHLA` 是否支持 prefix 预填充后的状态复用。
2. 递推时 suffix 的 position_ids 是否与训练保持一致。
3. cache 形式是 KV cache、递推状态，还是分块状态缓存。
4. 在线推理时是否仍然能保持动作质量稳定。


## 具体改动清单

建议按下面顺序实施：

### Step 1

在 `openpi` 新增 `mhla_joint_attention.py`，定义一个最小版本接口：

```python
att_output = mhla_joint_attention(
    query_states,
    key_states,
    value_states,
    attention_mask,
    position_ids,
    block_size=..., 
)
```


### Step 2

在 `gemma_pytorch.py` 的 `compute_layer_complete()` 中将：

```python
att_output, _ = modeling_gemma.eager_attention_forward(...)
```

改成可配置分支：

```python
if self.joint_attn_impl == "mhla":
    att_output = self.mhla_joint_attention(...)
else:
    att_output, _ = modeling_gemma.eager_attention_forward(...)
```


### Step 3

先做 hybrid 层替换：

- 例如只替换最后 `1/3` 或最后 `1/2` transformer layers
- 不建议第一版全层替换


### Step 4

验证训练路径：

- loss 是否正常下降
- 是否出现明显发散
- 显存占用是否下降
- 每 step latency 是否下降


### Step 5

再补推理路径中的 cache / recurrent 逻辑。


## 实验建议

### 第一组：正确性优先

- baseline：原始 `pi0`
- exp-1：最后 4 层替换为 MHLA
- exp-2：最后 8 层替换为 MHLA

关注：

- 训练 loss 曲线
- action MSE
- rollout 成功率


### 第二组：效率优先

- 统计 batch size 固定时的 step time
- 统计最大可训练序列长度
- 统计显存峰值


### 第三组：结构搜索

- 搜 `block_size`
- 搜替换层位置
- 搜 full vs hybrid


## 风险点

### 1. Mask 语义错位

这是最大风险。如果把 `pi0` 的 prefix-LM 结构错误近似成普通 causal attention，训练可能能跑，但动作质量会明显下滑。


### 2. Prefix / suffix 统计特性差异太大

图像 token、语言 token、state token、action token 的分布差异明显，统一套同一种 token grouping 可能不是最优。


### 3. 推理收益未必比训练收益大

`pi0` 推理本身已经对 prefix 做了 cache，在线推理瓶颈不一定完全在 full attention 上。因此 `MHLA` 更可能首先带来训练端收益。


### 4. 直接全层替换可能伤性能

建议从 hybrid 开始，不建议第一轮就 full replacement。


## 推荐的最小落地版本

如果只做一个最小、最现实的版本，我建议是：

1. 在 `openpi` 新增 `mhla_joint_attention.py`
2. 只改 `gemma_pytorch.py` 的联合 attention 主计算
3. 只替换最后若干层
4. 先只支持训练 forward
5. 跑通 loss、吞吐、显存对比后，再补推理 cache


## 一句话总结

对 `pi0` 而言，`MHLA` 不是一个“只增强 VLM”的插件，而是一个更适合接入 **联合 token mixer / joint self-attention** 的底层算子升级点。最稳妥的落地方式是：**先 hybrid、先训练、先改联合 attention，再逐步扩展到推理 cache。**
