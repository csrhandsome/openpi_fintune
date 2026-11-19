# 视触觉时空 token 与无 EMA 的 SIGReg 对照

日期：2026-10-28。官方参考源码克隆到 `third_party/jepa`、`third_party/vjepa2`、
`third_party/le-wm`，各自 `.UPSTREAM.md` 保存来源和 commit。

## 模型与数据

`config/train/force_encoder_jepa.yaml` 是唯一正式预训练配置；
`model.input_mode: single_frame | spatiotemporal` 控制 Conv2d／Conv3d。
默认视频模型每个样本 16 帧，片段内隔 2 帧取样，tubelet 2 帧，patch 16，224×224，
ViT-Base（768 维、12 层、12 heads），4 层 predictor。
batch 8 个片段，每个片段 1568 个 token，100k optimizer steps。
时间位置编码与二维空间位置编码相加，所有时空 tokens 一起参与 attention。
两类矩形 mask 各两个，在整个时间轴展开成 tube；所有目标从 context 中排除。
这是图像代码向视频的适配，不宣称逐项复现官方 V-JEPA 的完整训练 recipe。

每个候选片段都在同一个 episode、同一相机内，索引为
`start + [0,2,...,30]`；起点间隔为 `frame_stride: 5`。
片段长度不足不做尾帧 padding。仍按整个 episode 划分 90%／5%／5%，
按来源均衡有放回抽取片段。视频一次 seek 后顺序解码。

## 两种目标机制

`model.target_mode: ema` 保持冻结 EMA teacher 和归一化目标的 masked MSE。
`model.target_mode: sigreg` 只保留在线 encoder，完整输入分支和可见输入分支均回传
梯度，用 BN MLP projector 的目标计算 masked MSE，再加 SIGReg 正则。
正则作用于每个时间位置空间池化后的目标表征，population 是 batch 中独立片段；
不会把不同空间／时间位置当作独立样本而掩盖塌缩。

SIGReg 来自 [LeWorldModel 官方实现](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py)：
单位随机方向投影、17 个 `[0,3]` 频率节点、Gaussian characteristic function、
官方权重约定的 Epps–Pulley 统计，1024 个方向，weight 0.09。
FP32 计算统计量；固定输入、seed 的 CPU 对照中，损失和输入梯度与上游逐位相等。
许可保留于 `docs/licenses/le-wm-MIT.txt`。

该对照只迁移了 LeWM 的无 EMA／SIGReg 思路，预测任务仍是 masked video features；
不包含动作、因果 next-frame predictor 或规划器。仅凭训练 loss 不能证明表征质量。

## 训练、评测与追踪

```bash
PYTHONPATH=src .venv/bin/python -m force_encoder.train
PYTHONPATH=src .venv/bin/python -m force_encoder.train --target-mode sigreg \
  --exp-name jepa_video16_tube2_sigreg \
  --output-dir checkpoints/force_encoder/jepa_video16_tube2_sigreg
```

两者输出、SwanLab exp_name 和 run ID 独立。日志包含总 loss、prediction_loss、
sigreg_loss、跨样本 std、跨时间 std、每步帧数和显存，配置包含完整表示信息。
SIGReg 的 momentum 日志为 0。单帧历史实验 `jepa` 的进程与 checkpoint 保留。

CLI 支持 `--input-mode`、`--target-mode`、`--num-frames`、`--tubelet-size`、
`--exp-name`。有效配置写入各实验 checkpoint 目录的 `config.yaml`。
模型／数据配置变化时拒绝直接 resume；旧单帧权重仍能通过 `load_encoder` 导出。
时空推理要求同训练长度／帧间隔的有序片段；benchmark 的标签取片段末帧，
避免用片段未来图像去预测起点标签。

本机 `~/.ssh/config` 新增 `A6000`／`a6000`，保留 `mirnagroup` 和原地址别名。
配置变更前备份在 `/tmp/openpi-force-ssh-config.backup`。
同步只覆盖本模块、其配置及文档，使用 tar；远端覆盖前保存备份包。

## 验证

本地测试覆盖真实 MP4 的片段帧序、共享视频 episode 偏移、短 episode 排除、
recorded 时间戳、tube mask 的完整时间覆盖与隐藏像素梯度隔离、时间顺序敏感性、
两种模式的权重导出、SIGReg 目标端梯度与无 teacher、跨时间变化无法掩盖样本塌缩，
以及 single_frame／spatiotemporal × EMA／SIGReg 的 CPU 优化与中断恢复一致性。
本地与 A6000 全模块各 **70 passed**，Ruff 与格式检查通过。
另从 Git HEAD 加载修改前的单帧模型，对照初始化、loss 和全部参数梯度，逐位一致。

真实三源数据的 GPU 验收使用正式 Base、16 帧、batch 8、BF16，各完成 3 步优化、
一批验证及 best/latest 完整 checkpoint 保存：

| 模式 | allocated 峰值 GiB | reserved 峰值 GiB | 第 3 步总 loss | 验证总 loss |
| --- | ---: | ---: | ---: | ---: |
| EMA | 5.739 | 6.215 | 1.32464 | 1.33377 |
| SIGReg | 9.595 | 10.359 | 0.76782 | 0.67725 |

以上仅验证链路与显存，不代表预训练效果。两类目标的数值不可直接排序。
原始验收日志／配置／指标在远端 `outputs/force_encoder/video_smoke_20261028/`。
新视频 encoder 参数 86,236,416；EMA 训练参数 115,771,392。

已启动两个正式 100k 步实验，旧 `jepa` 保留：

| tmux 会话 | checkpoint 目录后缀 | SwanLab |
| --- | --- | --- |
| `jepa-video-ema` | `jepa_video16_tube2_ema` | [8bsqc7mp](https://swanlab.cn/@csrhandsome/openpi/runs/8bsqc7mp) |
| `jepa-video-sigreg` | `jepa_video16_tube2_sigreg` | [vjo3zohc](https://swanlab.cn/@csrhandsome/openpi/runs/vjo3zohc) |

两者首步已输出有限 loss 和梯度，SwanLab 已建立独立 cloud run；三任务共存时，
采样整卡占用约 25 GiB、GPU 利用率可达 100%（瞬时采样，不是长期均值）。
日志在 `outputs/force_encoder/jepa_video16_tube2_{ema,sigreg}.log`。
覆盖前备份：`outputs/force_encoder/deploy_video_20261028/before.tar.gz`。

连接与查看：

```bash
ssh A6000
tmux attach -t jepa-video-ema
tmux attach -t jepa-video-sigreg
```
