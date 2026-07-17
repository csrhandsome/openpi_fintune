# pi0.5 droid lora finetune script
# 先compute norm stats
export HF_LEROBOT_HOME="$(pwd)"
export HF_HUB_OFFLINE=1
uv run scripts/compute_norm_stats.py --config-name pi05_droid_lora_finetune
# 然后再fintune
uv run scripts/train.py pi05_droid_lora_finetune --exp_name pi5_2_24 --overwrite

# pi0 droid lora finetune script config-name是去查找config.py里对应的配置的。exp_name是给wandb的名字的 注意export的管理啊，如果tmux 里的不行，可以kill-session了之后开个新的
export HF_LEROBOT_HOME="$(pwd)"
# 这个参数，要是你要下载新的模型，就要是0，但是下载好了之后本地train，就要改成1
export HF_HUB_OFFLINE=1
# export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/compute_norm_stats.py --config-name pi0_base_audio_joint_lora --usad --usad-device cpu
uv run scripts/train.py pi0_base_audio_joint_lora --exp_name pi0_audio_6_1_usad --overwrite --usad --usad-device cpu

# aaai staged training
# Stage 0: route warmup -> ckpt_audio_route
# Stage 1: audio-action pretraining -> ckpt_audio_action
# Stage 2: mixed AA-AVLA training -> ckpt_mixed_avla
# Stage 3: full AVLA finetuning -> final_avla
uv run scripts/compute_norm_stats.py --config-name pi0_base_audio_joint_lora --usad --usad-device cpu
uv run scripts/train_stages.py \
  --exp-name aaai_audio_curriculum \
  --configs pi0_base_audio_joint_route_warmup_lora pi0_base_audio_joint_audio_action_lora pi0_base_audio_joint_mixed_aa_avla_lora pi0_base_audio_joint_full_avla_lora \
  --stage-exp-names ckpt_audio_route ckpt_audio_action ckpt_mixed_avla final_avla \
  --usad \
  --usad-device cpu \
  --overwrite

# 如果多线程ctrl c不能全部关掉用这个
pkill -KILL -f "scripts/train.py pi0_base_audio_joint_lora"
pkill -KILL -f "/home/three/code/openpi/.venv/bin/python3 -c from multiprocessing"