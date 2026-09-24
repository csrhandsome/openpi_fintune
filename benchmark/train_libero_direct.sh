#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/training"

: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${DATA_DIR:?Set DATA_DIR}"
: "${BASE_MODEL:?Set BASE_MODEL}"

accelerate launch --config_file './accelerator_config.yaml' \
  train_omni_libero_direct.py \
  --base_model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from_checkpoint "${RESUME_DIR:-${OUTPUT_DIR}}" \
  --data_root_dir "${DATA_DIR}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-150000}" \
  --data_mix "${DATA_MIX:-simple_audio_libero_spatial}" \
  --conditioning_mode "${CONDITIONING_MODE:-audio}" \
  --asr_baseline_manifest "${ASR_BASELINE_MANIFEST:-../data/libero_direct.json}" \
  --checkpoint_save_frequency "${CHECKPOINT_SAVE_FREQUENCY:-100}" \
  --future_action_window_size "${FUTURE_ACTION_WINDOW_SIZE:-5}" \
  --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE:-16000}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --logging_frequency "${LOGGING_FREQUENCY:-10}"
