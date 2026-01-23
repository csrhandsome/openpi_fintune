#!/bin/bash
# 网络诊断脚本（在服务器上运行）

echo "=========================================="
echo "服务器网络诊断"
echo "=========================================="

echo -e "\n[1] 主机名和 IP 地址:"
hostname
hostname -I

echo -e "\n[2] 网络接口详情:"
ip addr show | grep -E "^[0-9]+:|inet "

echo -e "\n[3] 监听端口 8000:"
netstat -tlnp 2>/dev/null | grep :8000 || ss -tlnp 2>/dev/null | grep :8000

echo -e "\n[4] 防火墙状态 (需要 sudo):"
echo "请手动运行: sudo ufw status verbose"

echo -e "\n[5] 从本地测试连接 8000 端口:"
timeout 2 bash -c "</dev/tcp/localhost/8000" 2>/dev/null && echo "✓ localhost:8000 可达" || echo "✗ localhost:8000 不可达"
timeout 2 bash -c "</dev/tcp/127.0.0.1/8000" 2>/dev/null && echo "✓ 127.0.0.1:8000 可达" || echo "✗ 127.0.0.1:8000 不可达"

echo -e "\n[6] 测试从本机网卡 IP 连接:"
for ip in $(hostname -I); do
    timeout 2 bash -c "</dev/tcp/$ip/8000" 2>/dev/null && echo "✓ $ip:8000 可达" || echo "✗ $ip:8000 不可达"
done

echo -e "\n[7] 路由表:"
ip route show

echo -e "\n=========================================="
echo "请将以上信息提供给调试使用"
echo "=========================================="
