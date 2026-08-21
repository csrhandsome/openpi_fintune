#!/usr/bin/env bash
# eval_talker_omni_object.sh — 从 ModelScope 拉 object Talker/Executor 权重,
# 然后用 interactive protocol 做 OmniAction-LIBERO object 冒烟评测。
#
# 流程:
#   1) 若本地 checkpoint 不完整,从 threecheng/livevla 下载 params + assets
#   2) 后台启动 policy server
#   3) 轮询端口直到 ready
#   4) 跑 scripts/run_benchmark.sh: libero_object × SPEECHES, protocol=interactive
#   5) 退出时 kill server,释放显存
#
# 用法:
#   bash scripts/eval_talker_omni_object.sh                         # 默认冒烟
#   SKIP_DOWNLOAD=1 bash scripts/eval_talker_omni_object.sh          # 已有权重
#   NUM_TRIALS_PER_TASK=10 SPEECHES=overlap,identity \
#     bash scripts/eval_talker_omni_object.sh                       # 更完整
#   PORT=8002 bash scripts/eval_talker_omni_object.sh
#
# 前置(本机常见缺口):
#   - scripts/modelscope/config.py 有 ACCESS_TOKEN(或 MODELSCOPE_API_TOKEN)
#   - 评测 wav: data/libero_audio/{overlap,identity,...}/
#       HF_HUB_OFFLINE=0 bash examples/omniaction_libero/download_data.sh --skip-rlds --skip-train-audio
#   - LIBERO 子模块 + eval venv(放在 benchmark/.venv):
#       git submodule update --init --recursive third_party/libero
#       uv venv --python 3.8 benchmark/.venv
#       source benchmark/.venv/bin/activate
#       uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
#         --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
#       uv pip install -e packages/openpi-client -e third_party/libero
#   - Qwen3-ASR-0.6B 已在 HF cache(离线加载)
#
# 显存: policy server(JAX pi0 + Qwen3-ASR)约需 ~12–18GB;LIBERO 仿真与 server 共用一张卡时
# 建议整卡空闲 ≥20GB。评测 client 可 unset CUDA 只用 CPU/EGL。
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

# --- ModelScope talker checkpoint ---
MS_REPO="${MS_REPO:-threecheng/livevla}"
MS_LOCAL_DIR="${MS_LOCAL_DIR:-$REPO_ROOT/checkpoints/threecheng_livevla}"
MS_MAX_WORKERS="${MS_MAX_WORKERS:-4}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/threecheng_livevla/checkpoints}"
CONFIG="${CONFIG:-pi0_omniaction_libero_object}"
CKPT="${CKPT:-$CHECKPOINT_ROOT/pi0_omniaction_libero_object/object_talker_executor_qwen32s/29999}"
# 远程 repo 内路径(snapshot 落到 MS_LOCAL_DIR 后与 CKPT 对齐)
REMOTE_CKPT_REL="${REMOTE_CKPT_REL:-checkpoints/pi0_omniaction_libero_object/object_talker_executor_qwen32s/29999}"
SUITE="libero_object"
STEM="${STEM:-object_talker_interactive}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
# 冒烟默认: 1 trial × overlap, interactive 协议
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
PROTOCOL="${PROTOCOL:-interactive}"
SPEECHES="${SPEECHES:-overlap}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
RESULT_DIR="${RESULT_DIR:-examples/omniaction_libero/results}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$RESULT_DIR/server_logs}"
VIDEO_ROOT="${VIDEO_ROOT:-data/omniaction_libero/videos/$STEM}"

mkdir -p "$RESULT_DIR" "$SERVER_LOG_DIR" "$LIBERO_CONFIG_PATH"

ckpt_ready() {
  local dir="$1"
  [[ -d "$dir/params" ]] || return 1
  find "$dir/assets" -name 'norm_stats.json' 2>/dev/null | grep -q . || return 1
  [[ "$(find "$dir" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ]] || return 1
  return 0
}

download_talker_ckpt() {
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "  -> SKIP_DOWNLOAD=1,跳过 ModelScope 下载"
    return 0
  fi
  if [[ "$FORCE_DOWNLOAD" != "1" ]] && ckpt_ready "$CKPT"; then
    echo "  -> 本地 checkpoint 已完整,跳过下载: $CKPT"
    return 0
  fi

  [[ -x "$POLICY_PYTHON" ]] || {
    echo "ERROR: POLICY_PYTHON 不可执行,无法下载: $POLICY_PYTHON" >&2
    exit 2
  }

  echo "  -> 从 ModelScope 下载 talker 权重"
  echo "     repo  = $MS_REPO"
  echo "     dest  = $MS_LOCAL_DIR"
  echo "     include = $REMOTE_CKPT_REL/{params,assets,_CHECKPOINT_METADATA}"
  mkdir -p "$MS_LOCAL_DIR"
  # ModelScope SDK 需要出网;评测阶段仍保持 HF_HUB_OFFLINE=1
  HF_HUB_OFFLINE=0 "$POLICY_PYTHON" scripts/modelscope/modelscope2weights.py "$MS_REPO" \
    --local-dir "$MS_LOCAL_DIR" \
    --include \
      "${REMOTE_CKPT_REL}/params/**" \
      "${REMOTE_CKPT_REL}/assets/**" \
      "${REMOTE_CKPT_REL}/_CHECKPOINT_METADATA" \
    --max-workers "$MS_MAX_WORKERS" || {
    echo "ERROR: ModelScope 下载失败" >&2
    exit 1
  }

  if ! ckpt_ready "$CKPT"; then
    echo "ERROR: 下载后 checkpoint 仍不完整: $CKPT" >&2
    echo "  期望目录: $MS_LOCAL_DIR/$REMOTE_CKPT_REL" >&2
    find "$CKPT" -name '*.incomplete' 2>/dev/null >&2 || true
    exit 1
  fi
  echo "  -> checkpoint 就绪: $CKPT"
}

# --- 确保 .libero 指向本仓库的 third_party/libero ---
ensure_libero_config() {
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
  if ! ckpt_ready "$CKPT"; then
    echo "ERROR: checkpoint 不完整: $CKPT" >&2
    echo "  需要 params/ 与 assets/*/norm_stats.json,且无 *.incomplete" >&2
    ok=0
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

download_talker_ckpt
preflight

echo "########################################################"
echo "# OmniAction-LIBERO object Talker interactive 评测"
echo "#   config   = $CONFIG"
echo "#   ckpt     = $CKPT"
echo "#   suite    = $SUITE"
echo "#   protocol = $PROTOCOL"
echo "#   speech   = $SPEECHES"
echo "#   host=$HOST port=$PORT trials/task=$NUM_TRIALS_PER_TASK"
echo "#   POLICY_PYTHON=$POLICY_PYTHON"
echo "#   EVAL_PYTHON=$EVAL_PYTHON"
echo "########################################################"

stop_server
pkill -f "serve_policy.py --port $PORT" 2>/dev/null || true
sleep 2

server_log="$SERVER_LOG_DIR/${STEM}_server.log"
echo "  -> 启动 server,日志: $server_log"
# Qwen on CPU keeps PyTorch off the JAX GPU. Combined with OPENPI_WARMUP=1,
# Talker JIT runs on the main thread before the websocket binds.
# --qwen-device is a top-level tyro flag and must precede policy:checkpoint.
export OPENPI_WARMUP="${OPENPI_WARMUP:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
QWEN_DEVICE="${QWEN_DEVICE:-cpu}"
setsid "$POLICY_PYTHON" scripts/serve_policy.py --port "$PORT" --qwen-device "$QWEN_DEVICE" \
  policy:checkpoint \
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
  echo ""
  echo "# interactive 对话样例:"
  grep -E "\[interactive/" "$bench_log" | head -20 || echo "  (未找到 interactive 日志行)"
else
  echo "  (无日志: $bench_log)"
fi
