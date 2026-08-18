#!/usr/bin/env bash
# Train the four RoboOmni-style OmniAction-LIBERO suite-specific models.
#
# Default: train object -> spatial -> goal -> 10 sequentially, using all
# available FSDP devices for each model. To train one suite only:
#   SUITES=object bash scripts/train_omniaction_libero_suites.sh
#
# Optional overrides:
#   FSDP_DEVICES=8 BATCH_SIZE=64 bash scripts/train_omniaction_libero_suites.sh
#   BATCH_SIZE=32 SUITES="object spatial" bash scripts/train_omniaction_libero_suites.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data/lerobot}"
export OMNIACTION_LIBERO_SIDECAR_ROOT="${OMNIACTION_LIBERO_SIDECAR_ROOT:-$REPO_ROOT/data/omniaction_libero/sidecars}"
export OMNIACTION_QWEN_FEATURE_CACHE_ROOT="${OMNIACTION_QWEN_FEATURE_CACHE_ROOT:-$REPO_ROOT/data/omniaction_libero/qwen_features}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || { echo "ERROR: TRAIN_PYTHON not executable: $TRAIN_PYTHON" >&2; exit 2; }

SUITES="${SUITES:-object spatial goal 10}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

for suite in $SUITES; do
  case "$suite" in
    object|spatial|goal|10) ;;
    *)
      echo "Unknown suite '$suite'. Valid suites: object spatial goal 10." >&2
      exit 2
      ;;
  esac

  config="pi0_omniaction_libero_${suite}"
  exp_name="${suite}_full_ft_qwen32s"
  stats="assets/${config}/omniaction_libero_${suite}/norm_stats.json"
  sidecar="$OMNIACTION_LIBERO_SIDECAR_ROOT/${suite}/audio"
  qwen_cache="$OMNIACTION_QWEN_FEATURE_CACHE_ROOT/omniaction_libero_${suite}"

  [[ -s "$stats" ]] || {
    echo "Missing norm stats: $stats" >&2
    exit 1
  }
  [[ -d "$sidecar" ]] || {
    echo "Missing audio sidecar directory: $sidecar" >&2
    exit 1
  }
  [[ -d "$qwen_cache" ]] || {
    echo "Missing Qwen feature cache: $qwen_cache" >&2
    echo "Build it with scripts/precompute_omniaction_qwen_features.py $config" >&2
    exit 1
  }

  echo "============================================================"
  echo "Training suite=$suite config=$config exp_name=$exp_name"
  echo "FSDP devices=$FSDP_DEVICES  batch size=$BATCH_SIZE"
  echo "sidecar=$sidecar"
  echo "qwen_cache=$qwen_cache"
  echo "============================================================"

  "$TRAIN_PYTHON" scripts/train.py "$config" \
    --exp_name "$exp_name" \
    --fsdp-devices "$FSDP_DEVICES" \
    --batch-size "$BATCH_SIZE" \
    --cashed \
    --overwrite
done