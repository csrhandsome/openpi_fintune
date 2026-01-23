# 官方droid
uv run scripts/serve_policy.py --env=DROID

# 查看realsense序列号
python vla-scripts/realworld/dual_camera_manager.py --list-devices

# franka 推理
python vla-scripts/realworld/franka_droid_main.py \
  --remote-host 192.168.5.89 \
  --remote-port 8000 \
  --franka-hostname 192.168.1.100 \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --joint-velocity-scale 0.5
