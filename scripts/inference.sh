# 官方droid
uv run scripts/serve_policy.py --env=DROID

# 查看realsense序列号
python vla-scripts/realworld/dual_camera_manager.py --list-devices

# franka 推理 本地4090 刚刚操作了一下老是通不了ws,后面发现是wsl比较特殊，需要在win上面把防火墙也打开，当win上面可以成功的访问到wsl的端口的时候，说明防火墙没有问题了，总的来说还是很多bug很麻烦。
uv run vla-scripts/realworld/franka_droid_main.py \
  --remote-host 192.168.5.38 \
  --remote-port 8001 \
  --franka-hostname 192.168.1.100 \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --joint-velocity-scale 1 \
  --color-only \
  --topk_samples 2

# joint 推理
uv run vla-scripts/realworld/franka_joint_main_audio.py \
    --remote-host 192.168.5.54 \
    --remote-port 8001 \
    --franka-hostname 192.168.1.100 \
    --external-camera-serial 825412070292 \
    --wrist-camera-serial 825412070487 \
    --color-only \
    --instruction "Place the object into the bassket" \
    --control-frequency-hz 13 \
    --open-loop-horizon 24 --rtc-inference-delay 8 \
    --max-joint-delta 0.1
    # rtc_inference_delay < open_loop_horizon
    # open_loop_horizon + rtc_inference_delay <= action_horizon

# franka 推理 5090
uv run vla-scripts/realworld/franka_droid_main.py \
  --remote-host 172.16.5.89 \
  --remote-port 8000 \
  --franka-hostname 172.16.5.89 \
  --external-camera-serial 825412070292 \
  --wrist-camera-serial 825412070487 \
  --joint-velocity-scale 1 \
  --color-only