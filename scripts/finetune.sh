# 先compute norm stats
export HF_LEROBOT_HOME="$(pwd)"
export HF_HUB_OFFLINE=1
uv run scripts/compute_norm_stats.py --config-name pi05_droid_lora_finetune

# 然后再fintune
uv run scripts/train.py pi05_droid_lora_finetune --exp_name openpi_fintune --overwrite