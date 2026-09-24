# OmniAction-LIBERO 训练与评测流程

本流程面向当前工作区：`/data/codes/grade_two/github/robot/openpi`。所有命令从项目根目录运行；策略服务与评测客户端统一使用 `127.0.0.1:8001`。路径均由 `REPO_ROOT` 推导，迁移目录时只需修改这一处。

## 运行环境

```bash
export REPO_ROOT=/data/codes/grade_two/github/robot/openpi
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
export no_proxy="0.0.0.0,127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
unset LEROBOT_HOME
```

使用项目主环境执行训练、server 和评测：`uv run ...`。不要为 Pi0Audio OmniAction config 添加 `--qwen`；Qwen3-ASR 已写入对应训练配置，额外 flag 会重置音频窗口设置。

## 数据与产物

| 内容 | 当前相对路径 / 位置 |
| --- | --- |
| 原始 RLDS 与训练语音 | `data/omniaction_libero/` |
| RoboOmni 评测 JSON | `third_party/RoboOmni/data/libero_*.json` |
| 评测 wav | `data/libero_audio/{overlap,identity,libero_sentiment,events,two,three}/` |
| LeRobot 数据集 | `$HF_LEROBOT_HOME/omniaction_libero_<suite>/` |
| 持久化 sidecar | `data/omniaction_libero/sidecars/<suite>/audio/` |
| norm stats | `assets/pi0_omniaction_libero_<suite>/omniaction_libero_<suite>/norm_stats.json` |
| checkpoint | `checkpoints/pi0_omniaction_libero_<suite>/<exp_name>/<step>/` |
| LIBERO 配置 | `.libero/config.yaml` |

`LeRobotOmniActionLiberoDataConfig` 当前读取 `data/omniaction_libero/sidecars/<suite>`。`/dev/shm` 可作为高速副本，但复制本身不会改写 config 的 `audio_sidecar_dir`；只有配置显式改为该目录时，训练才会读取内存盘。

## 1. 下载与构建数据

首次下载数据时关闭离线模式：

```bash
HF_HUB_OFFLINE=0 bash examples/omniaction_libero/download_data.sh
```

构建四个包含六种语音类型的 suite 数据集（CPU 执行，不占训练 GPU）：

```bash
CUDA_VISIBLE_DEVICES="" uv run examples/omniaction_libero/build_suite_all6.py --suite object
CUDA_VISIBLE_DEVICES="" uv run examples/omniaction_libero/build_suite_all6.py --suite spatial
CUDA_VISIBLE_DEVICES="" uv run examples/omniaction_libero/build_goal_all6.py
CUDA_VISIBLE_DEVICES="" uv run examples/omniaction_libero/build_suite_all6.py --suite 10
```

每次构建后检查对应的 LeRobot 数据集、`data/omniaction_libero/sidecars/<suite>/audio/episode_*.wav` 和 `.sync.json` 都存在。

## 2. 计算 norm stats

对每个将要训练的 config 计算一次：

```bash
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi0_omniaction_libero_object
```

将 `object` 替换为 `spatial`、`goal` 或 `10`。生成的 stats 必须位于对应 `assets/pi0_omniaction_libero_<suite>/omniaction_libero_<suite>/norm_stats.json`。

## 3. 训练

四个独立 suite 模型使用同一个入口；脚本会验证训练 sidecar 和对话标注：

```bash
SUITES=spatial FSDP_DEVICES=8 BATCH_SIZE=64 \
  bash scripts/train_omniaction_libero_suites.sh
```

完整训练四套模型：

```bash
SUITES="object spatial goal 10" bash scripts/train_omniaction_libero_suites.sh
```

脚本的当前 checkpoint 命名为 `<suite>_talker_executor_qwen32s`。如果改为手动训练，必须让 `--exp_name`、config 名称和训练 sidecar 路径保持一致。

三阶段 `object + overlap` 课程训练使用另一组 config：

```bash
uv run scripts/train_stages.py --exp-name omniaction_curriculum \
  --configs pi0_omniaction_libero_object_overlap_audio_action \
            pi0_omniaction_libero_object_overlap_mixed \
            pi0_omniaction_libero_object_overlap_full \
  --stage-exp-names ckpt_audio_action ckpt_mixed final_full --overwrite
```

三个 stage 共用 `omniaction_libero_object_overlap` 数据和其 norm stats；评测时的 `--policy.config` 必须精确对应最终加载的 stage。

## 4. 启动 policy server

以仓库内现有的 spatial checkpoint 为例：

```bash
uv run scripts/serve_policy.py --port 8001 policy:checkpoint \
  --policy.config=pi0_omniaction_libero_spatial \
  --policy.dir=checkpoints/threecheng_livevla/checkpoints/pi0_omniaction_libero_spatial/spatial_full_ft_qwen32s/29999
```

checkpoint 目录必须同时包含 `params/` 与 `assets/<asset_id>/norm_stats.json`。不要用通用 `--env LIBERO` 代替 OmniAction 的 checkpoint config。

## 5. 单个 suite 评测

另开终端，在 server 就绪后执行：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
unset CUDA_VISIBLE_DEVICES

uv run examples/omniaction_libero/eval_omniaction_libero.py \
  --task_suite_name libero_spatial \
  --speech overlap \
  --speech_json third_party/RoboOmni/data/libero_overlap.json \
  --audio_root data/libero_audio/overlap \
  --host 127.0.0.1 --port 8001 \
  --num_trials_per_task 10 \
  --video_out_path data/omniaction_libero/videos/spatial_overlap
```

如果所在节点要求其他 EGL renderer，显式覆盖 `MUJOCO_EGL_DEVICE_ID`。评测的 suite 必须和 server 上的 suite checkpoint 一致。更换语音时同时更新 `--speech`、`--speech_json` 和 `--audio_root`。

对于使用参考 robot dialogue prompt 训练的 Talker/Executor checkpoint，full-dialogue 评测加入 `--executor_prompt_mode reference`；旧 checkpoint 保持默认 `legacy_empty`。

## 6. 批量评测

已准备好对应 checkpoint 时，可由编排器串行启动 server、运行六种语音并回收进程：

```bash
SUITES=spatial PORT=8001 NUM_TRIALS_PER_TASK=10 \
  EVAL_PYTHON="$REPO_ROOT/.venv/bin/python" \
  bash scripts/run_all_eval.sh
```

单个已运行 server 的 suite 则使用：

```bash
SUITES=libero_spatial PORT=8001 \
  EVAL_PYTHON="$REPO_ROOT/.venv/bin/python" \
  bash scripts/run_benchmark.sh
```

日志默认写入 `examples/omniaction_libero/results/`，视频写入 `data/omniaction_libero/videos/`。

## 7. 运行前检查

1. `LIBERO_CONFIG_PATH` 指向 `$REPO_ROOT/.libero`，不能误用旧用户目录的 LIBERO 配置。
2. server config、checkpoint、评测 task suite 三者一致。
3. server 和 client 均为 `127.0.0.1:8001`。
4. Qwen3-ASR 权重已在 Hugging Face cache；离线模式下不能首次下载。
5. 每次评测会写入视频；根目录遗留视频已归档到 `data/videos/root_mp4/`。
