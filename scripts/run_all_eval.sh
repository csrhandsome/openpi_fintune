#!/usr/bin/env bash
# run_all_eval.sh — 依次评测四个 suite-specific OmniAction-LIBERO 权重。
#
# 每个 checkpoint 的流程(串行,一个接一个):
#   1) 后台启动 policy server(端口在 policy 加载完成后才 bind → 可用 TCP 连通性作 ready 信号)
#   2) 轮询端口直到 ready(Qwen 编码器加载较慢,超时可调 READY_TIMEOUT)
#   3) 跑 scripts/run_benchmark.sh,对该 suite × 全部 speech 评测
#   4) kill 掉 server 进程组,释放显存,再进入下一个 checkpoint
#
# 用法:
#   bash scripts/run_all_eval.sh                       # 默认只跑 libero_10 + full_dialogue
#   SUITES="object spatial goal 10" bash scripts/run_all_eval.sh
#   NUM_TRIALS_PER_TASK=20 bash scripts/run_all_eval.sh
#   PORT=8002 bash scripts/run_all_eval.sh
#   SUITES=object NUM_TRIALS_PER_TASK=1 bash scripts/run_all_eval.sh
#   SPEECHES="overlap,identity" bash scripts/run_all_eval.sh   # 只测部分 speech
#
# 注意:不是 set -e —— 单个 cell / 单个 checkpoint 失败不应中断整轮评测。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# --- server 侧环境(离线 + 本地 lerobot 缓存;qwen encoder 已 bake 进 config,无需 --qwen flag) ---
POLICY_PYTHON="${POLICY_PYTHON:-$REPO_ROOT/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$REPO_ROOT/benchmark/.venv/bin/python}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data/lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
[[ -x "$POLICY_PYTHON" ]] || { echo "ERROR: POLICY_PYTHON not executable: $POLICY_PYTHON" >&2; exit 2; }
[[ -x "$EVAL_PYTHON" ]] || { echo "ERROR: EVAL_PYTHON not executable: $EVAL_PYTHON" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export no_proxy="0.0.0.0,127.0.0.1,localhost"
export NO_PROXY="0.0.0.0,127.0.0.1,localhost"

# --- 可调参数 ---
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
# Whole-dialogue checkpoints consume the complete speech waveform per task.
PROTOCOL="${PROTOCOL:-full_dialogue}"
SUITES="${SUITES:-10}"
SPEECHES="${SPEECHES:-overlap,identity,libero_sentiment,events,two,three}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"   # 等 server ready 的最长秒数(Qwen 加载慢)
RESULT_DIR="${RESULT_DIR:-examples/omniaction_libero/results}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$RESULT_DIR/server_logs}"
VIDEO_ROOT="${VIDEO_ROOT:-data/omniaction_libero/videos}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/threecheng_livevla/checkpoints}"

mkdir -p "$RESULT_DIR" "$SERVER_LOG_DIR"

# --- 评测清单: 每行 "config|checkpoint_dir|eval_suite|log_stem" ---
# 每个 suite 只测本轮训练得到的 final checkpoint；每个模型跑全部 SPEECHES。
JOBS=(
  "pi0_omniaction_libero_object|$CHECKPOINT_ROOT/pi0_omniaction_libero_object/object_full_ft_qwen32s/29999|libero_object|object_full_ft"
  "pi0_omniaction_libero_spatial|$CHECKPOINT_ROOT/pi0_omniaction_libero_spatial/spatial_full_ft_qwen32s/29999|libero_spatial|spatial_full_ft"
  "pi0_omniaction_libero_goal|$CHECKPOINT_ROOT/pi0_omniaction_libero_goal/goal_full_ft_qwen32s/27000|libero_goal|goal_full_ft"
  "pi0_omniaction_libero_10|$CHECKPOINT_ROOT/pi0_omniaction_libero_10/10_full_ft_qwen32s/29999|libero_10|10_full_ft"
)

FILTERED_JOBS=()
for requested_suite in $SUITES; do
  case "$requested_suite" in
    object|spatial|goal|10) ;;
    *)
      echo "ERROR: Unknown suite '$requested_suite'. Valid suites: object spatial goal 10." >&2
      exit 2
      ;;
  esac
  for job in "${JOBS[@]}"; do
    if [[ "$job" == *"|libero_${requested_suite}|"* ]]; then
      FILTERED_JOBS+=("$job")
    fi
  done
done
JOBS=("${FILTERED_JOBS[@]}")

SERVER_PID=""

stop_server() {
  # kill 掉 server 及其整个进程组(setsid 启动 → PGID == SERVER_PID)
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  -> 停止 server (pid=$SERVER_PID)"
    kill -TERM "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "-$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

# 编排器被 Ctrl-C / 退出时,确保不残留 server 占着显存
trap 'echo; echo "== 收到中断,清理 server =="; stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

wait_ready() {
  local waited=0
  while (( waited < READY_TIMEOUT )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  !! server 进程已退出,详见日志"
      return 1
    fi
    if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
    if (( waited % 60 == 0 )); then
      echo "  ... 已等待 ${waited}s (仍在加载 policy)"
    fi
  done
  echo "  !! 等待 server ready 超时 (${READY_TIMEOUT}s)"
  return 1
}

echo "########################################################"
echo "# OmniAction-LIBERO 四个 suite 模型评测"
echo "#   实验数: ${#JOBS[@]}   host=$HOST port=$PORT trials/task=$NUM_TRIALS_PER_TASK"
echo "#   speech: $SPEECHES"
echo "#   结果目录: $RESULT_DIR"
echo "########################################################"

job_idx=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r config ckpt suite stem <<<"$job"
  job_idx=$((job_idx + 1))

  echo ""
  echo "========================================================"
  echo "==> [${job_idx}/${#JOBS[@]}] $stem"
  echo "    config = $config"
  echo "    ckpt   = $ckpt"
  echo "    suite  = $suite"
  echo "========================================================"

  if [[ ! -d "$ckpt" ]]; then
    echo "  !! checkpoint 目录不存在,跳过: $ckpt"
    continue
  fi

  # 保险:清掉可能残留在该端口上的旧 server
  stop_server
  pkill -f "serve_policy.py --port $PORT" 2>/dev/null || true
  sleep 2

  server_log="$SERVER_LOG_DIR/${stem}_server.log"
  echo "  -> 启动 server,日志: $server_log"
  # setsid: 独立进程组,便于整组 kill;</dev/null: 不占终端 stdin
  setsid "$POLICY_PYTHON" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
      --policy.config="$config" \
      --policy.dir="$ckpt" \
      > "$server_log" 2>&1 < /dev/null &
  SERVER_PID=$!

  if ! wait_ready; then
    echo "  !! server 未就绪,跳过 $stem"
    stop_server
    continue
  fi
  echo "  -> server ready,开始评测"

  bench_log="$RESULT_DIR/${stem}.log"
  HOST="$HOST" PORT="$PORT" \
  NUM_TRIALS_PER_TASK="$NUM_TRIALS_PER_TASK" \
  PROTOCOL="$PROTOCOL" \
  SUITES="$suite" SPEECHES="$SPEECHES" \
  VIDEO_ROOT="$VIDEO_ROOT/$stem" \
  LOG_FILE="$bench_log" \
  EVAL_PYTHON="$EVAL_PYTHON" \
  LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
  MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" \
    bash scripts/run_benchmark.sh || echo "  !! run_benchmark 返回非 0 (部分 cell 可能失败),继续"

  echo "  -> $stem 评测完成,结果: $bench_log"
  stop_server
  sleep 3
done

echo ""
echo "########################################################"
echo "# 全部评测完成。各 cell 成功率汇总:"
echo "########################################################"
for job in "${JOBS[@]}"; do
  IFS='|' read -r _config _ckpt _suite stem <<<"$job"
  bench_log="$RESULT_DIR/${stem}.log"
  echo ""
  echo "===== $stem ====="
  if [[ -f "$bench_log" ]]; then
    grep -E "TOTAL \[" "$bench_log" || echo "  (未找到 TOTAL 行,可能评测失败,见 $bench_log)"
  else
    echo "  (无日志: $bench_log)"
  fi
done