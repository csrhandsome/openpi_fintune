#!/usr/bin/env bash
# Evaluate one suite-specific OmniAction-LIBERO model across its six speech types.
#
# Prereq: policy server already running (terminal A), e.g.
#   .venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
#     --policy.config=pi0_omniaction_libero_10 \
#     --policy.dir=checkpoints/threecheng_livevla/checkpoints/pi0_omniaction_libero_10/10_full_ft_qwen32s/29999
#
# Usage:
#   SUITES="libero_object" bash scripts/run_benchmark.sh
#   SUITES="libero_goal" PORT=8001 NUM_TRIALS_PER_TASK=20 bash scripts/run_benchmark.sh
#   SUITES="libero_object" SPEECHES="overlap,identity" bash scripts/run_benchmark.sh
set -uo pipefail   # not -e: one cell failing should not abort the whole sweep

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# --- eval rendering env (override MUJOCO_EGL_DEVICE_ID if the sim GPU is not 0) ---
EVAL_PYTHON="${EVAL_PYTHON:-$REPO_ROOT/benchmark/.venv/bin/python}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$REPO_ROOT/.libero}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
unset CUDA_VISIBLE_DEVICES
[[ -x "$EVAL_PYTHON" ]] || { echo "ERROR: EVAL_PYTHON not executable: $EVAL_PYTHON" >&2; exit 2; }

# --- tunables ---
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
# Whole-dialogue checkpoints consume the complete speech waveform per task.
PROTOCOL="${PROTOCOL:-full_dialogue}"
VIDEO_ROOT="${VIDEO_ROOT:-data/omniaction_libero/videos}"
LOG_DIR="${LOG_DIR:-examples/omniaction_libero/results}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/suite_benchmark.log}"
SUITES="${SUITES:-}"
SPEECHES="${SPEECHES:-overlap,identity,libero_sentiment,events,two,three}"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"

if [[ -z "$SUITES" ]]; then
  echo "ERROR: Set exactly one suite, e.g. SUITES=libero_object bash scripts/run_benchmark.sh" >&2
  exit 2
fi
suite_count=$(wc -w <<<"$SUITES")
if [[ "$suite_count" -ne 1 ]]; then
  echo "ERROR: This runner serves one suite-specific model; got SUITES='$SUITES'." >&2
  exit 2
fi

# speech -> (speech_json stem, audio_root subdir)
declare -A JSON_STEM=(
  [overlap]=libero_overlap
  [identity]=libero_identity
  [libero_sentiment]=libero_sentiment
  [events]=libero_events
  [two]=libero_two
  [three]=libero_three
)
declare -A AUDIO_DIR=(
  [overlap]=overlap
  [identity]=identity
  [libero_sentiment]=libero_sentiment
  [events]=events
  [two]=two
  [three]=three
)

# Build the speech cells for this one suite.
IFS=',' read -ra _SPEECH_ARR <<<"$SPEECHES"
CELLS=()
for suite in $SUITES; do
  for speech in "${_SPEECH_ARR[@]}"; do
    CELLS+=("$suite|$speech")
  done
done

echo "==> Benchmarking ${#CELLS[@]} speech cells for $SUITES  host=$HOST port=$PORT trials=$NUM_TRIALS_PER_TASK"
echo "==> Videos -> $VIDEO_ROOT/<suite>_<speech>/   log -> $LOG_FILE"

idx=0
for cell in "${CELLS[@]}"; do
  IFS='|' read -r suite speech <<<"$cell"
  idx=$((idx + 1))
  json_stem="${JSON_STEM[$speech]}"
  audiodir="${AUDIO_DIR[$speech]}"
  vod="$VIDEO_ROOT/${suite}_${speech}"

  echo ""
  echo "==> [$idx/${#CELLS[@]}] $suite / $speech  (json=$json_stem  audio=$audiodir  video=$vod)"
  {
    echo "===== $suite / $speech ====="
    "$EVAL_PYTHON" examples/omniaction_libero/eval_omniaction_libero.py \
        --task_suite_name "$suite" \
        --speech "$speech" \
        --speech_json "third_party/RoboOmni/data/${json_stem}.json" \
        --audio_root "data/libero_audio/${audiodir}" \
        --video_out_path "$vod" \
        --host "$HOST" --port "$PORT" \
        --protocol "$PROTOCOL" \
        --num_trials_per_task "$NUM_TRIALS_PER_TASK"
  } 2>&1 | tee -a "$LOG_FILE" | grep -E "success=|TOTAL|success rate"
done

echo ""
echo "==> Done. Full log: $LOG_FILE"
echo "==> Per-cell success rates:"
grep -E "TOTAL \[" "$LOG_FILE" || true