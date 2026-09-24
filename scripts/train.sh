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
uv run scripts/train.py pi0_base_joint_lora_finetune --exp_name pi0_base_joint_lora_7_20 --overwrite --num_workers 4
uv run scripts/train.py pi0_base_joint_lora_finetune --exp_name pi0_base_joint_lora_7_20 --resume --num_workers 4

# 如果多线程ctrl c不能全部关掉用这个
pkill -KILL -f "/home/three/code/openpi/.venv/bin/python3 -c from multiprocessing"