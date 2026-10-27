# Force encoder：模型规模、显存与数据量评估

日期：2026-10-27。实测机器：`mirnagroup`，RTX A6000，CUDA 报告物理显存 **47.99 GiB**，WSL，PyTorch 2.7.1+cu128。

## 结论与正式配置

**结合现有数据和训练成本，首轮正式实验推荐 ViT-Base/16，单个 encoder 约 8565 万参数。** 这是一项有证据支持的起步选择，还没有通过长程下游实验确认最优。small（约 2159 万）和 large（约 3.03 亿）适合作为后续容量对照。当前不推荐仅为了占满显存而把十亿参数模型作为主实验。

**单看显存边界：** giant36 约 10.21 亿 encoder 参数，整卡采样占用约 42.34 GiB，保留约 12% 物理余量；giant 约 11.34 亿参数，占约 46.40 GiB，余量很少。giant48/wide 虽完成计算，却出现虚拟显存超配与明显变慢，不能算健康的训练配置。

配置已经收敛为两份，均位于 `config/train/`：

- [force_encoder_jepa.yaml](../config/train/force_encoder_jepa.yaml)：**默认已调整为 base**，batch 64、224×224、BF16，12 层 encoder、768 维、12 heads，4 层、768 维 predictor。
- [force_encoder_benchmark.yaml](../config/train/force_encoder_benchmark.yaml)：冻结 encoder 的下游评测。

临时 memory 扫描 YAML 及 small/base/large/giant48 的重复配置副本已从本地与远端清理，不长期维护多个规模 YAML。正式改变规模时修改同一份预训练配置的 `model` 字段。

## 当前从哪里初始化

当前入口默认**随机初始化，从头训练**。context encoder 随机初始化；target encoder 复制它，随后通过 EMA 更新；predictor 也随机初始化。没有加载 `third_party/tactile`、ImageNet 或其他外部预训练权重。本次每个规模的显存试验也是独立随机初始化，彼此不续训。

只有显式使用 `--resume /path/to/latest.pt` 才会恢复本模块的模型、EMA teacher、优化器、步数及已保存的随机状态。模型结构和数据配置必须匹配；原先 tiny 的 checkpoint 不能恢复成 base。启动日志会明确输出 `initialization=random` 或 `initialization=resume`。

原 `third_party/tactile` 的主 JEPA 输入为 taxel 时序，本模块输入为 RGB 图像，不能将其 checkpoint 当作当前图像模型直接继续训练。相同的是遮挡位置的特征预测、teacher/student 和 EMA 机制，不代表网络权重或完整实验 recipe 相同。

## 显存试验方法

固定 batch **64**、RGB **224×224**、patch **16**、BF16 autocast、FP32 参数、AdamW。encoder 和 predictor 一同扩宽，predictor 固定 4 层。未使用梯度累积、激活 checkpoint 或参数卸载。

从三份已核对的 NAS LeRobot v3 数据抽取相同的真实训练 batch 和验证 batch，各模型使用独立进程，完成 **8 次完整优化器更新**及验证；前 2 步作为预热。更新包含 teacher 前向、masked context/predictor 前向、MSE、反向传播、梯度裁剪、AdamW 和 EMA。优化器状态已经实际分配，不是只有 forward 的显存估计。

mask 在 32 个候选 seed 中选出 context token 数的最小、最大值，并与其他 mask 交替运行。这是有限采样压力测试，不是所有可能 mask 的数学上界。数据批次在本地缓存，因此以下步耗时是**纯计算测量**，不包含持续 NAS 读盘、保存 checkpoint 等成本；不应用它直接承诺正式训练时长。

## 实测结果

参数列只计算单个 encoder；训练时还包括 predictor、EMA teacher、梯度和优化器状态。所有档位均完成 8 次更新，但完成计算与健康地适配物理显存是两个不同结论。

| 规模 | encoder 参数（百万） | 维度 / 层数 / heads | allocated 峰值 GiB | reserved 峰值 GiB | 整卡占用采样峰值 GiB | 稳定步耗时 s | 判断 |
|---|---:|---|---:|---:|---:|---:|---|
| tiny | 5.49 | 192 / 12 / 3 | 1.15 | 1.35 | 2.98 | 0.039 | 低占用 |
| small | 21.59 | 384 / 12 / 3 | 2.46 | 2.77 | 4.40 | 0.072 | 低占用 |
| base | 85.65 | 768 / 12 / 12 | 5.75 | 6.29 | 7.91 | 0.171 | 推荐主实验起点 |
| large | 303.10 | 1024 / 24 / 16 | 13.99 | 15.18 | 16.81 | 0.443 | 建议作为容量对照 |
| huge | 630.66 | 1280 / 32 / 16 | 24.60 | 26.70 | 28.32 | 0.814 | 显存可用，收益尚未验证 |
| giant36（自定义） | 1021.12 | 1536 / 36 / 24 | 35.55 | 40.71 | 42.34 | 1.222 | 仅按显存、保留约 10% 余量的选择 |
| giant（自定义） | 1134.45 | 1536 / 40 / 24 | 38.97 | 44.77 | 46.40 | 1.340 | 高压力，余量约 3% |
| giant48（自定义） | 1361.10 | 1536 / 48 / 24 | 45.80 | 53.13 | 47.99 | 13.755 | 疑似超配/分页，不推荐 |
| wide（自定义） | 1543.72 | 1792 / 40 / 28 | 49.51 | 56.59 | 47.99 | 18.506 | 疑似超配/分页，不推荐 |

整卡占用来自阶段边界的 `cuda.mem_get_info` 采样，可能遗漏瞬时峰值。allocated/reserved 是 PyTorch allocator 指标；WSL/驱动环境中它们可能涉及虚拟分配，因此 reserved 超过物理容量不表示 GPU 真有额外物理显存。原始记录保留这些值，不将其截断或伪装为正常余量。

giant48/wide 的整卡占用触顶，同时 reserved 超过物理容量，步耗时比稍小模型大幅上升。这些现象提示超配或分页，本次没有额外采集驱动级 paging trace，故保留“疑似”表述；不能只因没有 OOM 就推荐它们。

本轮各档的训练峰值出现在 **forward 或邻近 backward 阶段**，不是保存 checkpoint。例如 large 的 forward 峰值约 13.99 GiB，AdamW 阶段约 7.85 GiB；giant 分别约 38.97 和 27.76 GiB。模型规模增长同时增加激活与常驻训练状态开销。

base 的精确参数量：encoder **85,646,592**；可训练的 context encoder + predictor **115,181,568**；含冻结 teacher 合计 **200,828,160**。以上显存试验不验证长期稳定性、收敛或下游表征质量。

## 数据规模与重复程度

仅纳入已核对的三份数据。完整数据合计 **8432 个 episodes、3,835,135 帧**。按来源内完整 episode 划分 90%/5%/5%，左右触觉相机作为独立图像；每 5 帧抽 1 帧。

| 来源 | 训练 episodes | 训练原始帧 | 可采训练图像（含左右相机） | 100k 步下每张图像的期望抽取次数 |
|---|---:|---:|---:|---:|
| tactile-video-pretrain-nc | 1863 | 1,762,698 | 706,528 | 3.02 |
| FreeTacMan | 5052 | 1,356,947 | 542,780 | 3.93 |
| ViTaMIn | 673 | 325,728 | 130,842 | 16.30 |
| 合计 | 7588 | 3,445,373 | **1,380,150** | 来源间不同 |

相邻帧与同 episode 的左右传感器存在相关性，因此 **138 万可采图像不等于 138 万独立场景**。本次未估算有效独立样本数，也未验证跨 episode 是否存在重复录制。转换后的 `task_index` 标签粒度还受转换器影响，不能直接用其条目数代表全部接触状态或物体多样性。

当前 100,000 步 × batch 64 = **640 万次图像抽取**。三源均衡、有放回采样，使每份数据期望各获得约 213 万次抽取，因此较小的 ViTaMIn 被重复得更多。640 万约为可采图像总数的 4.64 倍，但这不是 4.64 次完整遍历；`steps_per_epoch=1000` 是人为定义的日志/采样周期，不是遍历数据一次。

**100k 步只是初始计算预算，尚未验证足够。** 增大模型和重复同一批数据，不会自动增加训练数据多样性；训练时长与采样策略需要结合验证学习曲线再决定。

## 参数规模的建议与边界

1. **base，约 8565 万参数：主实验起点。** 显存约 8 GiB 整卡采样占用，计算明显轻于 large/huge，能先验证当前触觉图像及 JEPA recipe 是否学到可迁移信息。
2. **small 与 large：容量对照。** 固定数据版本、episode 划分、batch、采样预算及 head 评测协议，以验证集下游收益比较；先把训练不足与容量不足区分开。large 的显存约 17 GiB，已有实际扩容空间。
3. **huge/giant：在有收益证据后扩大。** 当前没有长程实验支持“模型越大越适合这批数据”，giant36 只是硬件余量结论，不能当作数据量匹配结论。

模型大小没有一个只看帧数就能确定的上限。I-JEPA 论文也在 ImageNet-1K 上训练了 Huge 模型，并研究增加数据多样性、模型规模的收益；这不意味着本次相关性更强的触觉视频也应该直接选 Huge。[I-JEPA 原论文](https://arxiv.org/html/2301.08243v3)。视觉模型扩展研究同样将数据、模型与计算预算联合考虑，而非只按显存选型。[Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)。这里的 base 推荐是针对当前成本与不确定性的判断，不是来自某条已验证的触觉 scaling law。

选择后续规模时，应观察真实下游标签上的验证集指标、泛化差距、表征退化迹象、GPU 小时与数据吞吐。不能只按 JEPA loss 下降选模型；teacher 目标本身也在变化。head seeds 与预训练 seeds 的不确定性分开报告，最终测试集保留到模型与 recipe 选择之后。

当前数据没有真实力标签；现有端到端验收使用 FreeTacMan 原 CSV 的夹爪距离，不能称为力估计精度。先前 32 步 tiny 预训练与 64 样本/3 epoch 的 probe 只能证明流程可运行，不能用来选择最终模型规模。

## 使用、记录与清理

远端项目：`/home/mirnagroup/code/openpi_fintune`。正式 base 从随机初始化启动：

```bash
cd /home/mirnagroup/code/openpi_fintune
PYTHONPATH=src .venv/bin/python -m force_encoder.train
```

目前没有启动正式 100k 步长程训练。原始诊断 JSON/日志保存在远端 `outputs/force_encoder/memory_sweep_20261027/`；[原始汇总 JSON](../results/force_encoder/memory_sweep_20261027/summary.json) 与 [诊断表](../results/force_encoder/memory_sweep_20261027/summary.md) 也保存于本地忽略版本控制的 `results/`，不包含图像数据。

临时诊断 YAML、各规模重复 YAML，以及本次诊断使用的真实图像 batch 缓存已清理；原始 NAS 数据未改动。保留通用诊断工具 `src/force_encoder/profile_memory.py`，以后如需重测必须显式传入临时配置，不再指向已删除的默认 memory YAML。

清理及默认规模调整后，模块本地与远端各 **54 项测试通过**，本地 Ruff 检查及格式检查通过。远端 GPU 已释放，确认没有留下正式长程训练进程。
