#!/usr/bin/env bash
# Robust OmniAction download via hf-mirror.
# Speech wavs use path-scoped listing (avoids full-repo tree -> 429).
set -euo pipefail
cd /home/three/code/openpi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
unset HF_DEBUG || true
export PYTHONUNBUFFERED=1
export HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

PY=(.venv/bin/python -u)
HF=("${PY[@]}" scripts/hf_with_mirror.py)
SUB=("${PY[@]}" scripts/hf_download_subdir.py)
LOG=/tmp/omniaction_download.log
exec > >(tee -a "$LOG") 2>&1

echo "==== $(date -Is) start ===="
echo "HF_ENDPOINT=$HF_ENDPOINT max_workers=$HF_MAX_WORKERS"
echo "TIP: if you just hit HTTP 429, wait 5-10 minutes before starting."

echo "==> [1/3] object RLDS (already-local files are skipped)"
"${HF[@]}" download OpenMOSS-Team/OmniAction-LIBERO --repo-type dataset \
  --include "libero_speech_object_*/**" \
  --local-dir data/omniaction_libero/rlds \
  --max-workers "$HF_MAX_WORKERS"

for sub in overlap events two three direct negative one_speaker semantic; do
  echo "==> [2/3] speech/$sub  $(date -Is)"
  # Path-scoped list+download: do NOT use `hf download --include` here
  # (that lists the entire OmniAction repo and triggers 429 on the mirror).
  "${SUB[@]}" OpenMOSS-Team/OmniAction \
    --repo-type dataset \
    --path "speech/${sub}" \
    --local-dir data/omniaction_libero \
    --max-workers "$HF_MAX_WORKERS"
  n=$(find "data/omniaction_libero/speech/$sub" -type f -name '*.wav' 2>/dev/null | wc -l || true)
  echo "    local wav count speech/$sub=$n"
  # Small pause between subdirs to ease mirror rate limits.
  sleep "${HF_SUBDIR_SLEEP:-3}"
done

echo "==> [3/3] eval wav  $(date -Is)"
"${HF[@]}" download OpenMOSS-Team/OmniAction-LIBERO-eval --repo-type dataset \
  --local-dir data/libero_audio \
  --max-workers "$HF_MAX_WORKERS"

echo "ALL_DONE $(date -Is)"
