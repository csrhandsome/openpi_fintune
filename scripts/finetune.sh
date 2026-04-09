# pi0.5 droid lora finetune script
# 先compute norm stats
export HF_LEROBOT_HOME="$(pwd)"
export HF_HUB_OFFLINE=1
uv run scripts/compute_norm_stats.py --config-name pi05_droid_lora_finetune
# 然后再fintune
uv run scripts/train.py pi05_droid_lora_finetune --exp_name pi5_2_24 --overwrite

# pi0 droid lora finetune script config-name是去查找config.py里对应的配置的。exp_name是给wandb的名字的
export HF_LEROBOT_HOME="$(pwd)"
export HF_HUB_OFFLINE=1
uv run scripts/compute_norm_stats.py --config-name pi0_audio_joint_lora
uv run scripts/train.py pi0_audio_joint_lora --exp_name pi0_audio_4_9 --overwrite