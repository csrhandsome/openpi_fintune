#!/usr/bin/env bash
# eval_omni_goal.sh — 单独评测本地 pi0_omniaction_libero_goal checkpoint。
#
# 流程(与 run_all_eval.sh 一致):
#   1) 后台启动 policy server
#   2) 轮询端口直到 ready
#   3) 跑 scripts/run_benchmark.sh: libero_goal × SPEECHES
#   4) 退出时 kill server,释放显存
#
# 用法:
#   bash scripts/eval_omni_goal.sh
#   NUM_TRIALS_PER_TASK=1 SPEECHES=overlap bash scripts/eval_omni_goal.sh   # 冒烟
#   SPEECHES="overlap,identity" PORT=8002 bash scripts/eval_omni_goal.sh
#   CKPT=checkpoints/threecheng_livevla/checkpoints/pi0_omniaction_libero_goal/goal_full_ft_qwen32s/27000 \
#     bash scripts/eval_omni_goal.sh
#
# 前置(本机常见缺口):
#   - checkpoint 完整(params/ 下无 *.incomplete),且含 assets/<asset_id>/norm_stats.json
#   - 评测 wav: data/libero_audio/{overlap,identity,...}/
#       HF_HUB_OFFLINE=0 bash examples/omniaction_libero/download_data.sh --skip-rlds --skip-train-audio
#   - LIBERO 子模块 + eval venv(放在 benchmark/.venv,不要放进 examples/libero 或 third_party/libero):
#       git submodule update --init --recursive third_party/libero
#       uv venv --python 3.8 benchmark/.venv
#       source benchmark/.venv/bin/activate
#       uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
#         --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
#       uv pip install -e packages/openpi-client -e third_party/libero
#   - .libero/config.yaml 用仓库根相对路径指向 third_party/libero(脚本会在缺失/路径不对时自动写一份)
#   - Qwen3-ASR-0.6B 已在 HF cache(离线加载)
#
# 显存: policy server(JAX pi0 + Qwen3-ASR)约需 ~12–18GB;LIBERO 仿真与 server 共用一张卡时
# 建议整卡空闲 ≥20GB(本机 4090 24GB 可行)。评测 client 可 unset CUDA 只用 CPU/EGL。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# --- python ---
POLICY_PYTHON="${POLICY_PYTHON:-$REPO_ROOT/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$REPO_ROOT/benchmark/.venv/bin/python}"

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data/lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export no_proxy="0.0.0.0,127.0.0.1,localhost"
export NO_PROXY="0.0.0.0,127.0.0.1,localhost"

# --- 本机 goal checkpoint(与 run_all_eval.sh 一致) ---
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/threecheng_livevla/checkpoints}"
CONFIG="${CONFIG:-pi0_omniaction_libero_goal}"
CKPT="${CKPT:-$CHECKPOINT_ROOT/pi0_omniaction_libero_goal/goal_full_ft_qwen32s/27000}"
SUITE="libero_goal"
STEM="${STEM:-goal_full_ft}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
PROTOCOL="${PROTOCOL:-full_dialogue}"
SPEECHES="${SPEECHES:-overlap,identity,libero_sentiment,events,two,three}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
RESULT_DIR="${RESULT_DIR:-examples/omniaction_libero/results}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$RESULT_DIR/server_logs}"
VIDEO_ROOT="${VIDEO_ROOT:-data/omniaction_libero/videos/$STEM}"

mkdir -p "$RESULT_DIR" "$SERVER_LOG_DIR" "$LIBERO_CONFIG_PATH"

# --- 确保 .libero 指向本仓库的 third_party/libero ---
ensure_libero_config() {
  # 相对仓库根目录;get_libero_path 按 CWD 解析,评测脚本都会 cd "$REPO_ROOT"
  local libero_rel="third_party/libero/libero/libero"
  local libero_root="$REPO_ROOT/$libero_rel"
  local cfg="$LIBERO_CONFIG_PATH/config.yaml"
  if [[ ! -d "$libero_root" ]]; then
    echo "WARNING: LIBERO 源码目录不存在: $libero_root" >&2
    echo "  请先: git submodule update --init --recursive third_party/libero" >&2
    return 0
  fi
  if [[ -f "$cfg" ]] && grep -q "$libero_rel" "$cfg" 2>/dev/null; then
    return 0
  fi
  cat > "$cfg" <<EOF
assets: ${libero_rel}/assets
bddl_files: ${libero_rel}/bddl_files
benchmark_root: ${libero_rel}
datasets: third_party/libero/datasets
init_states: ${libero_rel}/init_files
EOF
  echo "  -> 已写入 $cfg (相对路径指向本仓库 LIBERO)"
}

preflight() {
  local ok=1
  [[ -x "$POLICY_PYTHON" ]] || { echo "ERROR: POLICY_PYTHON 不可执行: $POLICY_PYTHON" >&2; ok=0; }
  [[ -x "$EVAL_PYTHON" ]] || {
    echo "ERROR: EVAL_PYTHON 不可执行: $EVAL_PYTHON" >&2
    echo "  按脚本头部注释安装 benchmark/.venv,或设置 EVAL_PYTHON=..." >&2
    ok=0
  }
  if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint 目录不存在: $CKPT" >&2
    ok=0
  else
    if [[ ! -d "$CKPT/params" ]]; then
      echo "ERROR: checkpoint 缺少 params/: $CKPT" >&2
      ok=0
    fi
    if ! find "$CKPT/assets" -name 'norm_stats.json' 2>/dev/null | grep -q .; then
      echo "ERROR: checkpoint 缺少 assets/*/norm_stats.json: $CKPT/assets" >&2
      echo "  serve_policy 加载时需要它;请补全该目录或从训练 assets 拷入。" >&2
      ok=0
    fi
    local incompletes
    incompletes="$(find "$CKPT" -name '*.incomplete' 2>/dev/null | wc -l)"
    if [[ "$incompletes" -gt 0 ]]; then
      echo "ERROR: checkpoint 含 $incompletes 个 *.incomplete 文件,权重可能未拷完:" >&2
      find "$CKPT" -name '*.incomplete' >&2
      ok=0
    fi
  fi

  declare -A _audio_dir=(
    [overlap]=overlap
    [identity]=identity
    [libero_sentiment]=libero_sentiment
    [events]=events
    [two]=two
    [three]=three
  )
  declare -A _json_stem=(
    [overlap]=libero_overlap
    [identity]=libero_identity
    [libero_sentiment]=libero_sentiment
    [events]=libero_events
    [two]=libero_two
    [three]=libero_three
  )
  local speech audio_dir js missing=0
  IFS=',' read -ra _speeches <<<"$SPEECHES"
  for speech in "${_speeches[@]}"; do
    audio_dir="${_audio_dir[$speech]:-}"
    js="${_json_stem[$speech]:-}"
    if [[ -z "$audio_dir" || -z "$js" ]]; then
      echo "ERROR: 未知 speech '$speech'" >&2
      ok=0
      continue
    fi
    if [[ -z "$(find "$REPO_ROOT/data/libero_audio/$audio_dir" -name '*.wav' 2>/dev/null | head -1)" ]]; then
      echo "ERROR: 缺少评测 wav: data/libero_audio/$audio_dir/*.wav" >&2
      missing=1
    fi
    if [[ ! -f "$REPO_ROOT/third_party/RoboOmni/data/${js}.json" ]]; then
      echo "ERROR: 缺少 speech json: third_party/RoboOmni/data/${js}.json" >&2
      ok=0
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "  下载: HF_HUB_OFFLINE=0 bash examples/omniaction_libero/download_data.sh --skip-rlds --skip-train-audio" >&2
    ok=0
  fi

  ensure_libero_config

  if [[ "$ok" -ne 1 ]]; then
    echo "preflight 失败,已中止。修好上述项后再跑。" >&2
    exit 2
  fi
}

SERVER_PID=""

stop_server() {
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

preflight

echo "########################################################"
echo "# OmniAction-LIBERO goal 单独评测"
echo "#   config = $CONFIG"
echo "#   ckpt   = $CKPT"
echo "#   suite  = $SUITE"
echo "#   speech = $SPEECHES"
echo "#   host=$HOST port=$PORT trials/task=$NUM_TRIALS_PER_TASK"
echo "#   POLICY_PYTHON=$POLICY_PYTHON"
echo "#   EVAL_PYTHON=$EVAL_PYTHON"
echo "########################################################"

stop_server
pkill -f "serve_policy.py --port $PORT" 2>/dev/null || true
sleep 2

server_log="$SERVER_LOG_DIR/${STEM}_server.log"
echo "  -> 启动 server,日志: $server_log"
setsid "$POLICY_PYTHON" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT" \
    > "$server_log" 2>&1 < /dev/null &
SERVER_PID=$!

if ! wait_ready; then
  echo "  !! server 未就绪,见 $server_log"
  exit 1
fi
echo "  -> server ready,开始评测"

bench_log="$RESULT_DIR/${STEM}.log"
HOST="$HOST" PORT="$PORT" \
NUM_TRIALS_PER_TASK="$NUM_TRIALS_PER_TASK" \
PROTOCOL="$PROTOCOL" \
SUITES="$SUITE" SPEECHES="$SPEECHES" \
VIDEO_ROOT="$VIDEO_ROOT" \
LOG_FILE="$bench_log" \
EVAL_PYTHON="$EVAL_PYTHON" \
LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" \
  bash scripts/run_benchmark.sh || echo "  !! run_benchmark 返回非 0 (部分 cell 可能失败)"

echo ""
echo "########################################################"
echo "# 评测完成。成功率汇总 ($bench_log):"
echo "########################################################"
if [[ -f "$bench_log" ]]; then
  grep -E "TOTAL \[" "$bench_log" || echo "  (未找到 TOTAL 行,见 $bench_log)"
else
  echo "  (无日志: $bench_log)"
fi
