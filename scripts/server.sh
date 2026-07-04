# pi0.5 的 server 启动脚本
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_lora_finetune \
  --policy.dir=checkpoints/pi05_droid_lora_finetune/pi5_2_22/19999

# pi0 的 server 启动脚本  可以在server_policy里面修改端口号
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_base_audio_joint_lora \
  --policy.dir=checkpoints/pi0_base_audio_joint_lora/pi0_audio_6_1_qwen/19999

# qwen版本的
uv run scripts/serve_policy.py --qwen --qwen-device cpu policy:checkpoint \
  --policy.config=pi0_base_audio_joint_lora \
  --policy.dir=checkpoints/pi0_base_audio_joint_lora/pi0_audio_6_1_qwen/19999

# 传递到5090(ssh)
rsync -avz --progress franka_droid_lerobot_20260130_161202/ \
    sustechdl@MS-7E06:/home/sustechdl/three/openpi_fintune/franka_droid_lerobot_20260130_161202/
