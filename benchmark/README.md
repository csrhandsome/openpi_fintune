# Direct Audio LIBERO Benchmark

This benchmark evaluates LIBERO manipulation with a simplified audio setting: each task uses one short spoken command. Direct audio-only runs use the constant text prompt `Follow the spoken instruction.`, so the target object or goal is supplied by audio rather than leaked through the text prompt.

## Data Manifest

- `data/libero_direct.json` is the direct benchmark manifest.
- `audio_path` points to the expected short-command WAV file.
- Audio files should be placed under `data/libero_audio/direct/<suite>/<task_id>.wav`.
- The audio folders are intentionally present as placeholders; generate or copy the new WAV files there before running evaluation.

## Evaluation

Audio-conditioned evaluation uses the clean command WAV plus the fixed text prompt:

```bash
python experiments/libero/run_libero_direct_eval.py \
  --pretrained_checkpoint MODEL_PATH \
  --task_suite_name libero_object \
  --speech direct \
  --conditioning_mode audio
```

ASR-baseline evaluation uses only the ASR transcript text from `asr_baseline_text` / `text_prompt` and does not pass audio:

```bash
python experiments/libero/run_libero_direct_eval.py \
  --pretrained_checkpoint MODEL_PATH \
  --task_suite_name libero_object \
  --speech direct \
  --conditioning_mode asr_baseline
```

The `libero_object` entries store clean TTS text in `tts_text` and degraded-ASR transcript text in `asr_baseline_text` / `text_prompt`; degraded audio is not kept as a benchmark input.

Supported task suites follow LIBERO naming: `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.

## Training

Train audio-conditioned and ASR-baseline models separately. Audio mode uses `speech_conv` audio with the fixed spoken-instruction text prompt. ASR-baseline mode uses image + ASR transcript text from the manifest and does not pass audio.

Audio-conditioned object training:

```bash
OUTPUT_DIR=... DATA_DIR=... BASE_MODEL=... \
DATA_MIX=simple_audio_libero_object \
AUDIO_ROOT_DIR=../data \
CONDITIONING_MODE=audio \
./train_libero_direct.sh
```

ASR-baseline object training:

```bash
OUTPUT_DIR=... DATA_DIR=... BASE_MODEL=... \
DATA_MIX=simple_audio_libero_object \
CONDITIONING_MODE=asr_baseline \
ASR_BASELINE_MANIFEST=../data/libero_direct.json \
./train_libero_direct.sh
```

Example mixture names reserved for direct RLDS builds:

- `simple_audio_libero_spatial`
- `simple_audio_libero_object`
- `simple_audio_libero_goal`
- `simple_audio_libero10`
- `simple_audio_libero_all`

## Attribution

This benchmark is a simplified derivative setup built on LIBERO task suites and code structure adapted from upstream evaluation/training utilities. Keep the applicable upstream licenses and citations when publishing a public release.
