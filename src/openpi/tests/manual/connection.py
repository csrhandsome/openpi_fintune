#!/usr/bin/env python3
"""
测试 OpenPI 策略服务器连接
"""

import socket
import sys
import time


def test_network_connectivity(host: str, port: int):
    """测试网络连通性"""
    print(f"\n{'=' * 60}")
    print("测试 1: 网络连通性测试")
    print(f"{'=' * 60}")
    print(f"目标: {host}:{port}")

    # 1. Ping 测试
    print("\n[1.1] Ping 测试...")
    import subprocess

    try:
        result = subprocess.run(["ping", "-c", "3", host], capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            print("✓ Ping 成功")
            # 提取延迟信息
            for line in result.stdout.split("\n"):
                if "time=" in line or "rtt" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ Ping 失败")
            print(f"  错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Ping 测试异常: {e}")
        return False

    # 2. TCP 连接测试
    print("\n[1.2] TCP 端口连接测试...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    try:
        start_time = time.time()
        sock.connect((host, port))
        connect_time = (time.time() - start_time) * 1000
        print(f"✓ TCP 连接成功 (耗时: {connect_time:.2f} ms)")
        sock.close()
        return True
    except TimeoutError:
        print("✗ 连接超时 (5秒)")
        print("  可能原因: 服务器未启动 或 防火墙阻止")
        return False
    except ConnectionRefusedError:
        print("✗ 连接被拒绝")
        print(f"  可能原因: 端口 {port} 上没有服务监听")
        return False
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        return False
    finally:
        sock.close()


def test_websocket_policy_server(host: str, port: int):
    """测试 WebSocket 策略服务器"""
    print(f"\n{'=' * 60}")
    print("测试 2: WebSocket 策略服务器测试")
    print(f"{'=' * 60}")

    try:
        from openpi_client import websocket_client_policy

        print("[2.1] 创建策略客户端...")
        policy_client = websocket_client_policy.WebsocketClientPolicy(host, port)
        print("✓ 客户端创建成功")

        print("\n[2.2] 测试连接...")
        # 创建一个虚拟观测数据
        import numpy as np

        test_observation = {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.array([0.0], dtype=np.float32),
            "prompt": "test connection",
        }

        print("  发送测试请求...")
        start_time = time.time()

        try:
            result = policy_client.infer(test_observation)
            inference_time = (time.time() - start_time) * 1000

            print(f"✓ 服务器响应成功 (耗时: {inference_time:.2f} ms)")
            print("\n  响应内容:")
            print(f"    - actions shape: {result.get('actions', 'N/A')}")
            if isinstance(result.get("actions"), np.ndarray):
                print(f"    - actions shape: {result['actions'].shape}")

            if "policy_timing" in result:
                print(f"    - policy_timing: {result['policy_timing']}")

            return True

        except Exception as e:
            print(f"✗ 推理请求失败: {e}")
            print("  可能原因: 服务器未正确启动或模型加载失败")
            return False

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("  请确保已安装 openpi-client")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_camera_connectivity(external_serial: str, wrist_serial: str):
    """测试相机连接"""
    print(f"\n{'=' * 60}")
    print("测试 3: RealSense 相机连接测试")
    print(f"{'=' * 60}")

    try:
        import pyrealsense2 as rs

        context = rs.context()
        devices = context.query_devices()

        print(f"[3.1] 检测到的相机数量: {len(devices)}")

        found_cameras = []
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"  [{i}] {name} (序列号: {serial})")
            found_cameras.append(serial)

        # 检查指定的相机
        print("\n[3.2] 验证指定的相机...")
        if external_serial:
            if external_serial in found_cameras:
                print(f"✓ 外部相机 {external_serial} 已连接")
            else:
                print(f"✗ 外部相机 {external_serial} 未找到")

        if wrist_serial:
            if wrist_serial in found_cameras:
                print(f"✓ 腕部相机 {wrist_serial} 已连接")
            else:
                print(f"✗ 腕部相机 {wrist_serial} 未找到")

        return len(devices) >= 2

    except ImportError:
        print("✗ pyrealsense2 未安装")
        print("  安装命令: pip install pyrealsense2")
        return False
    except Exception as e:
        print(f"✗ 相机测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print(f"\n{'#' * 60}")
    print("# OpenPI 连接诊断工具")
    print(f"{'#' * 60}")

    # 配置参数（从你的配置中读取）
    SERVER_HOST = "192.168.5.89"
    SERVER_PORT = 8000
    EXTERNAL_CAMERA_SERIAL = "825412070292"
    WRIST_CAMERA_SERIAL = "825412070487"

    print("\n配置信息:")
    print(f"  - 服务器地址: {SERVER_HOST}:{SERVER_PORT}")
    print(f"  - 外部相机序列号: {EXTERNAL_CAMERA_SERIAL}")
    print(f"  - 腕部相机序列号: {WRIST_CAMERA_SERIAL}")

    # 执行测试
    results = {}

    # 测试 1: 网络连通性
    results["network"] = test_network_connectivity(SERVER_HOST, SERVER_PORT)

    # 测试 2: WebSocket 策略服务器（仅在网络连通时测试）
    if results["network"]:
        results["policy_server"] = test_websocket_policy_server(SERVER_HOST, SERVER_PORT)
    else:
        results["policy_server"] = False
        print("\n⚠ 跳过策略服务器测试（网络不通）")

    # 测试 3: 相机连接
    results["cameras"] = test_camera_connectivity(EXTERNAL_CAMERA_SERIAL, WRIST_CAMERA_SERIAL)

    # 总结
    print(f"\n{'=' * 60}")
    print("测试总结")
    print(f"{'=' * 60}")

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name:20s}: {status}")
        all_passed = all_passed and passed

    print(f"\n{'=' * 60}")
    if all_passed:
        print("✓ 所有测试通过！可以开始运行机器人控制程序。")
        return 0
    print("✗ 部分测试失败，请根据上述错误信息排查问题。")
    print("\n常见问题排查:")
    if not results["network"]:
        print("  1. 检查服务器是否启动: uv run scripts/server.py --config config/train/openpi.yaml")
        print(f"  2. 检查服务器 IP 地址是否正确: {SERVER_HOST}")
        print(f"  3. 检查防火墙设置: sudo ufw allow {SERVER_PORT}")
    if not results["policy_server"]:
        print("  4. 检查策略服务器是否正常运行")
        print("  5. 查看服务器日志是否有错误")
    if not results["cameras"]:
        print("  6. 检查 RealSense 相机是否正确连接")
        print("  7. 检查相机序列号是否正确")
    return 1


if __name__ == "__main__":
    sys.exit(main())
