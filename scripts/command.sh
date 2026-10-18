# watch GPU usage every second
watch -n 1 nvidia-smi
# tumx
tmux new -s openvla
tmux attach -t openvla
tmux kill-session -t openvla  # 意思是：杀掉，目标是 openvla

# export代理
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
# 代理走我的电脑的7897端口
# 假设你电脑 IP 是 172.16.3.182 (请务必修改为你实际的 IP)
export http_proxy=http://172.16.3.182:7897
export https_proxy=http://172.16.3.182:7897
export all_proxy=socks5://172.16.3.182:7897
# 4090的设置（一个子网）
export http_proxy=http://192.168.5.18:7897
export https_proxy=http://192.168.5.18:7897
export all_proxy=socks5://192.168.5.18:7897

export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
# 清理到所有export出去的代理
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
env | grep -i proxy

# 一键关闭掉clash的代理
gsettings set org.gnome.system.proxy mode 'none' && unset http_proxy https_proxy all_proxy && pkill -f clash
# 如果没有输出（除了 grep 本身），说明进程已死。
ps aux | grep clash

export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 按 train.tracking_backend 选择对应平台登录（交互输入 API Key）
uv run swanlab login
# uv run wandb login

# hf镜像源
export HF_ENDPOINT=https://hf-mirror.com
