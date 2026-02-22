# pi0.5 的 server 启动脚本
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_droid_lora_finetune \
  --policy.dir=checkpoints/pi05_droid_lora_finetune/pi0_2_8/19999

# pi0 的 server 启动脚本  可以在server_policy里面修改端口号
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_droid_lora_finetune \
  --policy.dir=checkpoints/pi0_droid_lora_finetune/pi0_2_21/1000

# 传递到5090(ssh)
rsync -avz --progress franka_droid_lerobot_20260130_161202/ \
    sustechdl@MS-7E06:/home/sustechdl/three/openpi_fintune/franka_droid_lerobot_20260130_161202/
