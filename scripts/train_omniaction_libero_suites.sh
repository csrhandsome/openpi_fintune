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
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

SUITES="${SUITES:-object spatial goal 10}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

count_dialogue_turns() {
  python3 - "$1" <<'PY'
import json
import pathlib
import sys

audio_dir = pathlib.Path(sys.argv[1])
count = 0
for sync_path in audio_dir.glob("episode_*.sync.json"):
    try:
        data = json.loads(sync_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    dialogue = data.get("dialogue")
    if isinstance(dialogue, list):
        count += sum(
            isinstance(turn, dict) and bool(str(turn.get("robot_response", "")).strip())
            for turn in dialogue
        )
print(count)
PY
}

for suite in $SUITES; do
  case "$suite" in
    object|spatial|goal|10) ;;
    *)
      echo "Unknown suite '$suite'. Valid suites: object spatial goal 10." >&2
      exit 2
      ;;
  esac

  config="pi0_omniaction_libero_${suite}"
  exp_name="${suite}_talker_executor_qwen32s"
  stats="assets/${config}/omniaction_libero_${suite}/norm_stats.json"
  persistent_audio="omniaction_libero_${suite}/audio"
  sidecar="/dev/shm/omniaction_libero_${suite}/audio"

  [[ -s "$stats" ]] || {
    echo "Missing norm stats: $stats" >&2
    exit 1
  }
  [[ -d "$persistent_audio" ]] || {
    echo "Missing persistent audio sidecar directory: $persistent_audio" >&2
    exit 1
  }

  expected_sync_count="$(find "$persistent_audio" -maxdepth 1 -type f -name 'episode_*.sync.json' | wc -l)"
  [[ "$expected_sync_count" -gt 0 ]] || {
    echo "No dialogue sync files found under persistent sidecars: $persistent_audio" >&2
    exit 1
  }
  expected_dialogue_turn_count="$(count_dialogue_turns "$persistent_audio")"
  [[ "$expected_dialogue_turn_count" -gt 0 ]] || {
    echo "No supervised robot dialogue turns found under: $persistent_audio" >&2
    exit 1
  }

  # /dev/shm is intentionally used for fast audio I/O but is volatile across
  # node/session restarts. A previous copy can have the correct number of
  # episode files while still containing stale .sync.json metadata without
  # dialogue annotations, so refresh every file rather than only checking
  # counts.
  echo "Refreshing audio sidecars in /dev/shm for suite=$suite"
  echo "  source=$persistent_audio ($expected_sync_count sync files, $expected_dialogue_turn_count dialogue turns)"
  mkdir -p "$sidecar"
  cp -af "$persistent_audio/." "$sidecar/"

  current_sync_count="$(find "$sidecar" -maxdepth 1 -type f -name 'episode_*.sync.json' | wc -l)"
  current_dialogue_turn_count="$(count_dialogue_turns "$sidecar")"
  [[ "$current_sync_count" -eq "$expected_sync_count" ]] || {
    echo "Audio sidecar sync verification failed for $suite: expected $expected_sync_count sync files, found $current_sync_count under $sidecar" >&2
    exit 1
  }
  [[ "$current_dialogue_turn_count" -eq "$expected_dialogue_turn_count" ]] || {
    echo "Dialogue sidecar sync verification failed for $suite: expected $expected_dialogue_turn_count robot turns, found $current_dialogue_turn_count under $sidecar" >&2
    exit 1
  }

  echo "============================================================"
  echo "Training suite=$suite config=$config exp_name=$exp_name"
  echo "FSDP devices=$FSDP_DEVICES  batch size=$BATCH_SIZE"
  echo "============================================================"

  uv run --offline scripts/train.py "$config" \
    --exp_name "$exp_name" \
    --fsdp-devices "$FSDP_DEVICES" \
    --batch-size "$BATCH_SIZE" \
    --cashed \
    --overwrite
done