# 视触觉 JEPA encoder

独立的 PyTorch 训练、数据读取和推理实现，不依赖 `third_party`。默认使用 ViT-Base/16，输入是视触觉传感器的 RGB 图像，输出为 768 维 pooled embedding 或 `196 × 768` patch tokens。规模建议与完整显存诊断见 [容量评估报告](../../docs/force_encoder_capacity_2026_10_27.md)。

本模块训练的是可供力估计等任务复用的触觉表征；当前预训练没有使用力标签，不直接输出牛顿单位的力值。默认 encoder 为 12 层、768 维、12 heads，predictor 为 4 层、768 维。采用 EMA target encoder，以及 2 个局部和 2 个全局矩形 target mask。context 严格排除所有 target patch，MSE 回归归一化的 teacher features。图像共用同一份 resize 和 ImageNet normalization，teacher 始终处于 eval 模式。这是针对图像的 JEPA 实现，原有 taxel 拓扑图 mask 不适用于 RGB patch。

## 远端数据与环境

mirnagroup 项目实际路径是 `/home/mirnagroup/code/openpi_fintune`。下面的命令均从项目根目录执行，使用已有 `.venv`，不需要安装 `third_party/tactile`。依赖为 `torch`、`numpy`、`Pillow`、`pyarrow`、`av` 和 `pyyaml`；测试还需要 `pytest`。

```bash
cd /home/mirnagroup/code/openpi_fintune
PYTHONPATH=src .venv/bin/python -m force_encoder.setup_data --nas-root /mnt/nas
```

这只创建 `data/force -> /mnt/nas`，不会替换已有文件或改动 NAS 数据。配置统一位于根目录 `config/train`：[预训练](../../config/train/force_encoder_jepa.yaml)、[下游测评](../../config/train/force_encoder_benchmark.yaml)。数据路径相对于项目根目录解析：

```text
data/force/tactile-video-pretrain-nc/lerobot_v3
data/force/FreeTacMan/lerobot_v3
data/force/ViTaMIn/lerobot_v3
```

只接受 `v3.0` 且具有匹配 `verified.json` 的数据。WireFishing-M、HapTile 和 FeelSight 不在默认配置里；转换并核对完毕后再显式添加。默认仅选择 `observation.gripper_image_left/right`，不读取环境或腕部相机；左右传感器作为独立图像样本，共用 encoder。以整个 episode 划分 90%/5%/5% 训练、验证、测试集，同一 episode 的左右图像始终在同一 split；测试 episode 不参加 JEPA 预训练。按数据集均衡采样，默认每 5 帧取 1 帧。

**时间轴兼容：** NAS 上第一份数据的 `info.fps=1`、Parquet 时间戳也是 1 FPS，而 MP4 实为 30 FPS；FreeTacMan 元数据是 31 FPS，MP4 为 30 FPS。默认使用帧序号和 MP4 真实帧率定位，并考虑 v3 episode 视频起始偏移，避免按这些错误时间戳错读。未来时间戳正确且不等间隔的数据可使用 recorded 模式（见 `data.py`）。读取器不写数据、不修复原始元信息、不自动下载。

## 检查和训练

检查全部元数据并实际解码每份数据的一张触觉图像（仅打印结构和数值检查）：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.train --inspect
```

运行默认单 GPU 配置：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.train
```

默认 batch size 64，BF16，100,000 optimizer steps，每 1,000 步验证并保存。每个 epoch 为 1,000 次均衡采样的 batch，并非遍历整个数据集。输出位于 `checkpoints/force_encoder/jepa/`，包括 `config.yaml`、split/数据规模信息、`metrics.jsonl`、`latest.pt` 和 `best.pt`。日志中的 feature 标准差用于观察表征是否退化，短程 loss 下降并不能证明预训练表征有效。

不传 `--resume` 时 encoder 随机初始化，teacher 复制 encoder；不会加载原项目或 ImageNet 的预训练权重。启动日志明确打印 `initialization=random` 或 `resume`。默认已从初始 tiny 调整为 base，旧 tiny checkpoint 不能用于 base 的恢复训练，但仍可独立加载推理。

短程验证：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.train \
  --max-steps 20 --batch-size 16 --num-workers 2 --val-batches 2 \
  --output-dir checkpoints/force_encoder/smoke
```

恢复训练需使用相同数据、seed、模型和训练计划；测试时可以调整 `--max-steps` 来检查恢复入口，但正式实验应保留原有 schedule：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.train \
  --resume checkpoints/force_encoder/jepa/latest.pt
```

当前入口支持单 GPU；没有启动多卡 DDP。数据解码采用 worker 本地有界缓存，避免为数百万帧创建 Python 索引列表。

## 与原 JEPA 的对应关系和显存

这是图像输入的适配，并非原项目实验的逐项复现。下面比较原项目 `config/xela/pretrain/jepa.yaml` 的 tiny 配置：

| 项目 | 原 Xela JEPA | 当前图像 JEPA |
|---|---|---|
| 目标 | masked taxel embedding 的 MSE | masked image patch embedding 的 MSE |
| encoder | 192 维、12 层、3 heads | 默认 base：768 维、12 层、12 heads |
| predictor | 4 层 | 4 层 |
| 输入 | 368 taxels × 3 通道 × 10 帧 | 单张 RGB 224×224，16×16 patch，共 196 tokens |
| mask | taxel 图连通/分组采样 | 图像矩形，2 local + 2 global |
| batch / GPU | 每卡 64，4 卡，总 batch 256 | 单卡 64 张独立图像，总 batch 64 |
| 位置编码 / dropout | learned / drop path 0.3 | 固定二维正余弦 / dropout 0 |
| EMA 起点 | 0.994 | 0.996 |
| learning rate | 1e-4 | 3e-4 |
| weight decay 终值 | 0.4 | 0.1 |
| warmup / 时长 | 30 epochs / 500 epochs | 2,000 steps / 100,000 steps |
| 梯度裁剪 | 10 | 1 |

当前 teacher 保持 eval 且额外进行 token feature normalization；原默认 teacher 跟随训练状态、未打开额外 target normalization。原项目的 `deco/pretrain/jepa_wd01` 是另一套 recipe，也不能与 Xela 原配置混称为完全对齐。当前每个 batch 样本是一张传感器图像，没有将多帧或双传感器拼成同一个序列。

2026-10-27 在 RTX A6000（48 GiB）上，初始 tiny 用 batch 64 连续训练 32 步并验证成功。随后统一 batch 64、BF16、同一真实数据批次扫描 9 个规模，每档完成 8 次完整更新与验证；默认 base 的 encoder 为 **85,646,592** 参数，训练参数（context encoder + predictor）为 **115,181,568**，含 EMA teacher 共 **200,828,160** 参数。base 峰值 allocated 为 **5.748 GiB**、reserved 为 **6.287 GiB**，阶段边界采样整卡占用约 **7.915 GiB**。详细表格及超配诊断统一保存在 docs 容量报告，日志保存 `cuda_*_gib` 指标。

因此当前配置的 batch 64 已实测可用。训练步数增加的是运行时间和总采样次数，计算图在每次 backward 后释放，不会把每一步的显存累计起来。100,000 步仍是待执行的长程计划，不等于已经跑完，也不等价于论文 500 epochs。

## 下游 benchmark

`benchmark.py` 提供冻结 EMA encoder 的监督回归/分类，支持 linear/MLP head 和 frozen-random encoder 基线。它读取真实 Parquet 标签，先提取特征到磁盘 memmap，再只训练 head；回归均值/标准差只由训练标签计算，预测恢复到原始单位。每个 head seed 用验证集选择最佳 checkpoint，之后只在独立测试集评估一次。

新 benchmark 要求预训练 checkpoint 也预留同一份 test split；旧的 95%/5% smoke checkpoint 会被拒绝，防止把预训练已见过的数据作为独立测试。split seed、比例、来源目录必须保持一致。固定版本的数据应保持原地不可变；增加数据后应建立新版本并重新预训练和评估。

默认测评 **FreeTacMan 的夹爪距离**，不是力：已检查转换器，该字段来自原 CSV 的 `gripper_distance`，转换后放在 `observation.gripper_position`，并未转换为开合比例。源单位未由转换器说明，因此保留 `source_csv_distance`，不假设为米或比例。`tactile-video-pretrain-nc` 的同名字段在本次转换中是零占位，不能用于有效夹爪测评。三份数据目前都没有可用于力回归的标签。

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.benchmark
```

默认使用正式预训练 `checkpoints/force_encoder/jepa/best.pt`，训练 3 个 seed 的 linear head，输出到 `outputs/force_encoder/gripper_probe/`。如需更换任务，修改 `config/train/force_encoder_benchmark.yaml` 的 `task`：

- 位移回归：`type: regression`、`target_key: observation.ee_position`、`label_names: [x, y, z]`，使用来源确认的单位。直接对 `ee_pose` 的角度算普通误差不等价于 SE(3) 位姿指标。
- 任务类别分类：`type: classification`、`target_key: task_index`，删除默认回归 `label_names/units`，从训练集建立类别表；`task_index` 是数据集内任务 ID，不等于自动获得物体或动作类别。不同数据集的本地 task ID 不能直接混合。
- 力回归：未来数据存在真实力列后，将 `target_key` 指向该列，并设置匹配的维度名称与单位；缺少标签会直接报错，不生成假标签。

同样划分和 head seeds 下的冻结随机 encoder 基线：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.benchmark \
  --baseline frozen_random --output-dir outputs/force_encoder/gripper_random
```

每个输出目录包含 `config.yaml`、逐 split 的 `features/` 缓存、各 `seed_N/` 的 `best_head.pt`、训练日志、`test_metrics.json` 和 `evaluation/{predictions.npz,manifest.json}`。artifact 记录样本/episode 身份、数据 fingerprint、checkpoint hash、标签语义、单位、split 协议和 seed。`load_head()` 支持读取最佳监督 head。

统计与报告入口：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.evaluation.report \
  --runs outputs/force_encoder/gripper_probe/seed_0 \
         outputs/force_encoder/gripper_probe/seed_1 \
         outputs/force_encoder/gripper_probe/seed_2 \
  --baseline outputs/force_encoder/gripper_random/seed_0 \
  --bootstrap-samples 2000 --permutation-samples 2000 \
  --output outputs/force_encoder/report.json
```

会生成 JSON 和 Markdown。回归指标为 RMSE、MAE、逐维指标和 R²（常量维度的 R² 为 null）；分类为 accuracy、balanced accuracy、macro F1 和 confusion matrix。按 source/episode 聚类 bootstrap，保留左右传感器及帧之间的相关性。配对比较严格对齐样本、标签、单位、数据版本和划分，并提供 group permutation 和 Holm 校正。以上单个 baseline 命令只对匹配的 seed 0 做配对比较；seed 1/2 的对应比较需各自指定相同 seed 的 baseline。多 seed 的均值/标准差与测试 episode 的置信区间分开报告。

这些是新图像 encoder 的下游工具；原项目的 Xela/Socks/DECO 具体任务需要各自传感器输入和标签，不能把这里的夹爪/任务分类称为论文 force/object/DECO policy benchmark。短程、小样本测评仅验收流程，不能证明预训练效果优于随机 encoder。

## 推理

加载 EMA encoder，输入需要 RGB、resize 到 224×224，并用 ImageNet mean/std 归一化：

```python
from force_encoder.model import load_encoder

encoder = load_encoder("checkpoints/force_encoder/jepa/best.pt", device="cuda")
embedding = encoder(images)                    # 默认 base: [B, 768]
tokens = encoder.forward_features(images)      # 默认 base: [B, 196, 768]
```

也可直接从图片文件导出 NumPy embeddings：

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.infer \
  --checkpoint checkpoints/force_encoder/jepa/best.pt \
  --images /path/to/tactile.png --output /tmp/tactile_embedding.npy
```

添加 `--tokens` 导出 patch tokens。训练 checkpoint 同时保存 context/target encoder、predictor、优化器、步数和 RNG 状态；`load_encoder` 只加载 EMA encoder。

## 本地开发与同步

在本地修改后，只同步模块，不覆盖远端其他项目文件：

```bash
rsync -av --exclude __pycache__ --exclude .pytest_cache \
  src/force_encoder/ mirnagroup:/home/mirnagroup/code/openpi_fintune/src/force_encoder/
rsync -av config/train/force_encoder_*.yaml \
  mirnagroup:/home/mirnagroup/code/openpi_fintune/config/train/
PYTHONPATH=src .venv/bin/python -m pytest src/force_encoder/tests -q
```

## 已完成的远端验证（2026-10-27）

在 mirnagroup 的 RTX A6000 上使用全部三份已核对数据，保持默认 224×224、12 层 encoder 和 4 层 predictor：

- batch 16 / 2 workers：20 步训练 + 验证成功，loss 从首步 `1.38701` 到第 20 步 `1.33913`；验证 loss `1.33912`。
- 从 checkpoint 恢复并继续至第 22 步，验证 loss `1.32125`。该恢复测试调整了总步数，只用于验收恢复入口，不作为训练效果实验。
- 默认 batch 64 / 4 workers / BF16：3 步训练 + 验证 + 保存成功，进程正常退出。
- 从真实验证图像加载 EMA encoder，API 与图片 CLI 均成功导出 `[1,192]` 全局特征和 `[1,196,192]` patch 特征，数值有限。
- 本地与远端各 19 项测试通过；本地 Ruff 检查通过。

远端测试产物位于 `checkpoints/force_encoder/smoke_20261027/`（22 步）和 `checkpoints/force_encoder/batch64_clean_20261027/`（默认 batch 验证）。正式配置的输出目录 `checkpoints/force_encoder/jepa/` 留给长程训练。以上为流程验收，尚未进行 100,000 步的表征质量实验。

后续增加独立测试划分与下游测评后的最终验收：

- `checkpoints/force_encoder/memory64_holdout_20261027/`：新 90%/5%/5% 协议，batch 64 训练 32 步并验证，峰值 allocated 1.158 GiB，reserved 1.371 GiB。该 checkpoint 可用于下游流程测试；上面旧 95%/5% checkpoint 只能继续做推理等检查。
- `outputs/force_encoder/benchmark_jepa_final_20261027/`、`benchmark_random_smoke_20261027/`：两种 encoder 各运行 3 个 head seeds；每个 split 选 64 张真实样本，每个 head 训练 3 epochs，保存了最佳 head、测试预测及指标。
- `outputs/force_encoder/benchmark_all_seeds_20261027.{json,md}`：两种 encoder 的多 seed 汇总。夹爪距离 RMSE 为 JEPA `28.2024 ± 0.1434`、随机 `28.2047 ± 0.1387`（原 CSV 单位，± 为 head seed 间标准差）。短程预训练及少量 head 更新的结果接近，只证明测评流程能运行。
- `outputs/force_encoder/benchmark_report_20261027.{json,md}`：seed 0 的配对比较和 JEPA 多 seed 汇总；`benchmark_paired_seed_1_20261027`、`benchmark_paired_seed_2_20261027` 为相应 seed 的配对报告。使用 512 次 cluster bootstrap 和 512 次 group permutation。
- 新版本本地与远端各 **38 项测试通过**，本地 Ruff 检查及格式检查通过。

后续已完成 9 档规模诊断和显存超配识别，完整单元测试增至 54 项。默认改为 base，临时扫描和各规模副本 YAML 已清理；长期仅保留预训练及下游测评两个正式配置。诊断详情及基于数据量的建议以 docs 容量报告为准。
