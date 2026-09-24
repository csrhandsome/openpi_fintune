#!/usr/bin/env bash
# Download OmniAction-LIBERO training RLDS, training speech wavs, and eval audio.
#
# Defaults (can override via env):
#   HF_ENDPOINT=https://hf-mirror.com
#   HF_HUB_DISABLE_XET=1
#   HF_MAX_WORKERS=4
#
# Usage:
#   bash examples/omniaction_libero/download_data.sh                # all suites
#   bash examples/omniaction_libero/download_data.sh --suite object  # one suite
#   bash examples/omniaction_libero/download_data.sh --skip-train-audio
#   bash examples/omniaction_libero/download_data.sh --skip-eval
#   bash examples/omniaction_libero/download_data.sh --skip-rlds
#   # or the simpler mirror entrypoint:
#   bash scripts/download_omniaction_mirror.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default to China-friendly mirror + plain HTTPS (override by exporting before run).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
unset HF_DEBUG 2>/dev/null || true

SUITE="all"
SKIP_RLDS=0
SKIP_TRAIN_AUDIO=0
SKIP_EVAL=0

RLDS_REPO="OpenMOSS-Team/OmniAction-LIBERO"
SPEECH_REPO="OpenMOSS-Team/OmniAction"
EVAL_REPO="OpenMOSS-Team/OmniAction-LIBERO-eval"

RLDS_DIR="${ROOT}/data/omniaction_libero/rlds"
SPEECH_DIR="${ROOT}/data/omniaction_libero"
EVAL_DIR="${ROOT}/data/libero_audio"

# RLDS speech-type suffixes under libero_speech_*_
SPEECH_TYPES=(events sentiment overlap identity three two)
# OmniAction training wav subdirs under speech/ (verified against Hub tree)
SPEECH_SUBDIRS=(overlap events two three direct negative one_speaker semantic)

usage() {
  cat <<'EOF'
Download OmniAction-LIBERO assets via `hf download`.

Options:
  --suite {spatial,object,goal,10,all}   RLDS suite filter (default: all)
  --skip-rlds                           Skip OmniAction-LIBERO RLDS
  --skip-train-audio                    Skip OmniAction speech/ wav tree
  --skip-eval                           Skip OmniAction-LIBERO-eval wavs
  -h, --help                            Show this help

Env (optional overrides; defaults are set in this script):
  HF_ENDPOINT       Hub endpoint (default: https://hf-mirror.com)
  HF_MAX_WORKERS    Parallel download workers (default: 4)
  HF_SUBDIR_SLEEP   Pause between speech subdirs (default: 3)
EOF
}

suite_prefixes() {
  case "$1" in
    spatial) echo "libero_speech_spatial_" ;;
    object) echo "libero_speech_object_" ;;
    goal) echo "libero_speech_goal_" ;;
    10) echo "libero_speech10_" ;;
    all) echo "" ;;
    *)
      echo "Unknown suite: $1 (expected spatial|object|goal|10|all)" >&2
      exit 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --skip-rlds)
      SKIP_RLDS=1
      shift
      ;;
    --skip-train-audio)
      SKIP_TRAIN_AUDIO=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Prefer mirror-aware wrapper (pagination Link headers from hf-mirror still
# point at huggingface.co; wrapper rewrites them).
HF_WRAPPER="${ROOT}/scripts/hf_with_mirror.py"
if [[ -f "${HF_WRAPPER}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    HF_BIN=("${ROOT}/.venv/bin/python" "${HF_WRAPPER}")
  else
    HF_BIN=(python3 "${HF_WRAPPER}")
  fi
elif [[ -x "${ROOT}/.venv/bin/hf" ]]; then
  HF_BIN=("${ROOT}/.venv/bin/hf")
elif command -v hf >/dev/null 2>&1; then
  HF_BIN=("$(command -v hf)")
else
  echo "ERROR: \`hf\` CLI not found. Install with: uv pip install -U huggingface_hub" >&2
  exit 1
fi
echo "Using hf: ${HF_BIN[*]}"
echo "Using HF_ENDPOINT=${HF_ENDPOINT} max_workers=${HF_MAX_WORKERS}"

MAX_RETRIES="${HF_DOWNLOAD_RETRIES:-8}"

hf_download_retry() {
  local attempt=1
  local delay=5
  while true; do
    if "${HF_BIN[@]}" download "$@"; then
      return 0
    fi
    if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
      echo "ERROR: hf download failed after ${MAX_RETRIES} attempts: $*" >&2
      return 1
    fi
    echo "WARN: hf download failed (attempt ${attempt}/${MAX_RETRIES}); retry in ${delay}s..." >&2
    sleep "${delay}"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
    if [[ "${delay}" -gt 120 ]]; then
      delay=120
    fi
  done
}

PREFIX="$(suite_prefixes "${SUITE}")"

if [[ "${SKIP_RLDS}" -eq 0 ]]; then
  mkdir -p "${RLDS_DIR}"
  if [[ -z "${PREFIX}" ]]; then
    echo "==> Downloading full RLDS: ${RLDS_REPO} -> ${RLDS_DIR}"
    hf_download_retry "${RLDS_REPO}" --repo-type dataset --local-dir "${RLDS_DIR}"
  else
    echo "==> Downloading RLDS suite=${SUITE} (prefix=${PREFIX}*) -> ${RLDS_DIR}"
    for speech in "${SPEECH_TYPES[@]}"; do
      pattern="${PREFIX}${speech}/**"
      echo "  include ${pattern}"
      hf_download_retry "${RLDS_REPO}" \
        --repo-type dataset \
        --include "${pattern}" \
        --local-dir "${RLDS_DIR}"
    done
  fi
fi

if [[ "${SKIP_TRAIN_AUDIO}" -eq 0 ]]; then
  mkdir -p "${SPEECH_DIR}"
  echo "==> Downloading training speech wavs: ${SPEECH_REPO}/speech -> ${SPEECH_DIR}"
  # Path-scoped download avoids full-repo list_repo_tree (which 429s on hf-mirror).
  SUB_DL=("${ROOT}/.venv/bin/python" "${ROOT}/scripts/hf_download_subdir.py")
  if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
    SUB_DL=(python3 "${ROOT}/scripts/hf_download_subdir.py")
  fi
  for sub in "${SPEECH_SUBDIRS[@]}"; do
    echo "  path speech/${sub}"
    "${SUB_DL[@]}" "${SPEECH_REPO}" \
      --repo-type dataset \
      --path "speech/${sub}" \
      --local-dir "${SPEECH_DIR}" \
      --max-workers "${HF_MAX_WORKERS:-4}"
    sleep "${HF_SUBDIR_SLEEP:-3}"
  done
fi

if [[ "${SKIP_EVAL}" -eq 0 ]]; then
  mkdir -p "${EVAL_DIR}"
  echo "==> Downloading eval wavs: ${EVAL_REPO} -> ${EVAL_DIR}"
  hf_download_retry "${EVAL_REPO}" \
    --repo-type dataset \
    --local-dir "${EVAL_DIR}"
fi

echo "Done."
echo "  RLDS:         ${RLDS_DIR}"
echo "  train audio:  ${SPEECH_DIR}/speech"
echo "  eval audio:   ${EVAL_DIR}"
