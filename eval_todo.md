# OmniAction-LIBERO 评测与数据准备

## 目标与当前结论

目标权重：

```
checkpoints/threecheng_livevla/checkpoints/pi0_omniaction_libero_10/10_full_ft_qwen32s/29999
```

它对应 staged 配置 `pi0_omniaction_libero_10`。已实际启动过 policy server：checkpoint 已进入 Orbax 参数恢复阶段，没有出现文件缺失、参数 key 或 shape 不匹配。

但 checkpoint 约 29GB；本机只有一张 8GB RTX 4070 Laptop GPU。恢复过程中申请额外 2.25GB 即 OOM，server 无法绑定端口。因此此机器不能完成真实 policy 推理或整套评测。

## 已完成的路径修复

脚本不再依赖旧机器上的 `~/DataFromRoot/.../openpi`、`/dev/shm` 或 GPU 8。当前默认值：

| 项 | 现在的默认 |
| --- | --- |
| 仓库根 | 由脚本位置推导 (`REPO_ROOT`) |
| Policy server Python | `$REPO_ROOT/.venv/bin/python` |
| LIBERO 客户端 Python | `$REPO_ROOT/benchmark/.venv/bin/python` |
| Checkpoint | `checkpoints/threecheng_livevla/checkpoints/pi0_omniaction_libero_<suite>/<suite>_full_ft_qwen32s/...` |
| 协议 | `full_dialogue` |
| EGL 设备 | `MUJOCO_EGL_DEVICE_ID=0` |
| LIBERO 配置 | `$REPO_ROOT/.libero` |
| LeRobot 数据 | `$REPO_ROOT/data/lerobot` |
| 训练 sidecar | `$REPO_ROOT/data/omniaction_libero/sidecars/<suite>/` |
| Qwen 特征缓存 | `$REPO_ROOT/data/omniaction_libero/qwen_features/` |
| 训练入口 | 直接调用 `.venv` 中的 Python，不再 `uv run --offline` |

评测启动（whole-dialogue audio，suite-10）：

```bash
POLICY_PYTHON="$PWD/.venv/bin/python" \
EVAL_PYTHON="$PWD/benchmark/.venv/bin/python" \
LIBERO_CONFIG_PATH="$PWD/.libero" \
MUJOCO_EGL_DEVICE_ID=0 \
SUITES=10 PROTOCOL=full_dialogue \
NUM_TRIALS_PER_TASK=10 \
bash scripts/run_all_eval.sh
```

当前 GPU 显存不足时，该命令会在恢复 checkpoint 时 OOM；这不是权重格式或音频数据问题。

已验证以下评测输入均存在且匹配：

- `libero_10` 的 10 个任务与六种 speech JSON 对齐。
- `data/libero_audio/{overlap,identity,libero_sentiment,events,two,three}` 的 eval wav 均可解析。
- 根目录 `.venv` 在 `LIBERO_CONFIG_PATH=$PWD/.libero` 和 EGL GPU 0 下已完成一次 LIBERO 单步渲染。
- 离线 Qwen3-ASR processor 缓存可加载。

## norm stats：本次实测

已执行等价于下列命令的 CPU-only norm-stats 计算：

```bash
JAX_PLATFORMS=cpu \
HF_LEROBOT_HOME="$PWD/data/lerobot" \
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_omniaction_libero_10
```

实际错误为：

```
FileNotFoundError: No parquet episodes found under
.../omniaction_libero_10/data
```

原因是当前仓库只有 RoboOmni 的 task JSON 和评测音频，没有 RoboOmni 的 RLDS trajectory，也没有转换后的 LeRobot parquet。仅靠 JSON 和 wav 不含 `state/actions`，不能计算训练用 norm stats。

## 应先构建 suite-10 训练数据

是的，应使用 `examples/omniaction_libero/build_suite_all6.py`。它会：

1. 读取 suite-10 的 identity 与 sentiment RLDS trajectory。
2. 使用任务 instruction 匹配，为 events/overlap/two/three 合成对应的 audio-action 配对。
3. 生成 LeRobot parquet 与 per-episode audio sidecar。
4. 然后才能计算 norm stats。

仍需补齐的原始数据：

```
data/omniaction_libero/libero_speech10_identity/**/1.0.0/*.tfrecord*
data/omniaction_libero/libero_speech10_sentiment/**/1.0.0/*.tfrecord*
```

数据就绪后的顺序：

```bash
CUDA_VISIBLE_DEVICES="" \
HF_LEROBOT_HOME="$PWD/data/lerobot" \
.venv/bin/python examples/omniaction_libero/build_suite_all6.py --suite 10

JAX_PLATFORMS=cpu \
HF_LEROBOT_HOME="$PWD/data/lerobot" \
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi0_omniaction_libero_10
```

builder 的 sidecar 应放在仓库内，例如：

```
data/omniaction_libero/sidecars/10/audio/
```

## 当前环境还缺什么

### 仅评测已有 checkpoint

- 足够显存的 policy-server GPU；本机 8GB 已验证不足。
- 如果使用 `benchmark/.venv`，需要将 Torch 1.11 升级至兼容当前 LIBERO 的 Torch 2.x。当前 LIBERO 调用 `torch.load(..., weights_only=False)`，Torch 1.11 会报错。
- 评测客户端需要能导入 `openpi.shared.dialogue`；`run_benchmark.sh` 已把 `$REPO_ROOT/src` 加入 `PYTHONPATH`。

评测不依赖新构建的 training parquet 或 sidecar：目标 checkpoint 自身已带有 `assets/omniaction_libero_10/norm_stats.json`，eval JSON 和 wav 也已经齐全。路径硬编码已按上表改完。

### 构建或训练新权重

- 上述 identity/sentiment RLDS tfrecord。
- builder 生成的 `data/lerobot/omniaction_libero_10/data/*.parquet`。
- builder 生成的本地 sidecar：`data/omniaction_libero/sidecars/10/audio/`。
- compute 后的 `assets/pi0_omniaction_libero_10/omniaction_libero_10/norm_stats.json`。
- 使用 `--cashed` 训练时所需的本地 Qwen 特征缓存：`data/omniaction_libero/qwen_features/omniaction_libero_10/`。
- 适合 FSDP 的多 GPU 显存；当前单张 8GB GPU 不适合训练。

## 代码是否已足够，还是只差数据？

评测代码链路已经足够：whole-dialogue waveform 输入、任务/音频映射、policy client、LIBERO 单步仿真均已验证。当前评测不是只差数据，主要还差可恢复该 checkpoint 的 GPU 显存，以及 benchmark 环境的 Torch 版本。

训练链路则不仅缺一个 norm-stats 文件：首先缺原始 RLDS trajectory。获取后可用 builder 构建 parquet 和 sidecar，再计算 norm stats；之后还需要 Qwen 特征缓存和足够的训练显存。

## 32 秒音频窗口

当前模型的 instruction-audio 窗口为 32 秒。部分 whole-dialogue 更长：events 最长约 62.5 秒，three 的样本全部超过 32 秒。现有变换会裁剪输入；这不是缺文件，但会影响严格的完整对话语义。
