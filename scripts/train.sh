# pi0.5 droid lora finetune script
# 先compute norm stats
export HF_LEROBOT_HOME="$(pwd)"
export HF_HUB_OFFLINE=1
uv run scripts/compute_norm_stats.py --config-name pi05_droid_lora_finetune
# 然后再fintune
uv run scripts/train.py pi05_droid_lora_finetune --exp_name pi5_2_24 --overwrite

export HF_LEROBOT_HOME="$(pwd)"
# 这个参数，要是你要下载新的模型，就要是0，但是下载好了之后本地train，就要改成1
uv run scripts/compute_norm_stats.py --config-name pi0_base_joint_lora_finetune
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
# 限制cpu的线程
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
uv run scripts/train.py pi0_base_joint_lora_finetune --exp_name pi0_audio_7_20_asr --overwrite --num_workers 4
uv run scripts/train.py pi0_base_joint_lora_finetune --exp_name pi0_audio_7_20_asr --resume --num_workers 4



# pi0 droid lora finetune script config-name是去查找config.py里对应的配置的。exp_name是给wandb的名字的 注意export的管理啊，如果tmux 里的不行，可以kill-session了之后开个新的
export HF_LEROBOT_HOME="$(pwd)"
# 这个参数，要是你要下载新的模型，就要是0，但是下载好了之后本地train，就要改成1
uv run scripts/compute_norm_stats.py --config-name pi0_base_audio_joint_lora
export HF_HUB_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
# 限制cpu的线程
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
uv run scripts/train.py pi0_base_audio_joint_lora --exp_name pi0_audio_7_20_usad --overwrite --usad --usad-device cpu --num_workers 3
uv run scripts/train.py pi0_base_audio_joint_lora --exp_name pi0_audio_7_20_usad --resume --usad --usad-device cpu --num_workers 3
uv run scripts/train.py pi0_base_audio_joint_lora --exp_name pi0_audio_7_20_qwen --overwrite --qwen --qwen-device cpu --num_workers 3
uv run scripts/train.py pi0_base_audio_joint_lora --exp_name pi0_audio_7_20_qwen --resume --qwen --qwen-device cpu --num_workers 3


# 如果多线程ctrl c不能全部关掉用这个
pkill -KILL -f "scripts/train.py pi0_base_audio_joint_lora"
pkill -KILL -f "/home/three/code/openpi/.venv/bin/python3 -c from multiprocessing"