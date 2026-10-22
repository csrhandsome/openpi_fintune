# Triton 推理开关与 A6000 验证

在 `config/server/openpi.yaml` 或 `config/server/openpi_rlt.yaml` 中设置：

```yaml
use_triton: true
device: cuda
```

默认 `use_triton: false`。统一服务入口的 VLA、RLT 和 force 模式均传递该开关，也支持
`python scripts/server.py --config config/server/openpi.yaml --use-triton True`。

开启后，Gemma 的 RMSNorm 和 Pi0.5 的 adaRMS 使用 Triton 融合内核，累加保留 FP32，
输出保持原始 dtype。只有 CUDA、eval、关闭 autograd 时执行融合路径；训练和求梯度沿用
原来的实现。开启时传入 CPU 模型会明确报错。参数、checkpoint 键和模型权重均不改变。
首次遇到新的形状或 dtype 会编译内核，首次延迟不代表稳态性能。

沿用现有环境：`torch==2.7.1+cu128` 在 Linux x86_64 下已依赖 Triton，`uv.lock` 已锁定
`triton==3.3.1`。本次不修改 `pyproject.toml` 和 `uv.lock`，不需要额外安装包。
原有 GELU、RoPE 等 `torch.compile` 路径在开关两种状态下均保持原设置。

## 远端测试

2026-10-22，在 `mirnagroup:/home/mirnagroup/code/openpi_fintune` 的 NVIDIA RTX A6000
上验证。PyTorch 2.7.1+cu128、Triton 3.3.1；真实 checkpoint 为
`checkpoints/openpi/joint_7_6_audio_qwen_asr/30000`，Pi0 BF16、LoRA rank 48、动作维度
32、预测长度 50、Euler 10 步。

```bash
JAX_PLATFORMS=cpu SWANLAB_MODE=disabled .venv/bin/python -m pytest \
  src/openpi/tests/models_pytorch/triton_inference_test.py \
  src/openpi/tests/models_pytorch/pi0_pytorch_test.py \
  src/openpi/tests/serving \
  src/openpi/tests/rlt/force_rlt/yaml_stage_test.py -q
```

结果：**57 passed**。`JAX_PLATFORMS=cpu` 仅限制辅助 JAX 初始化；Triton 数值与缓存测试
使用 CUDA。测试覆盖 FP32/BF16/FP16、普通及 adaptive RMSNorm、批次和非整齐宽度、
autograd/训练回退、前缀与缓存后缀、配置校验以及 VLA/force 的开关传递。SwanLab 在
测试进程中关闭上报，避免依赖在线日志服务。

## 完整推理基准

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_triton.py \
  --config config/server/openpi.yaml \
  --warmup 5 --repeats 15 --rounds 4 \
  --output /tmp/openpi-triton-benchmark-final.json
```

同一个已加载模型、合成 Franka 双相机图像与状态、固定噪声。计时覆盖 `Policy.infer`
的预处理、模型和动作后处理，调用前后同步 CUDA；每种模式预热 5 次，分 4 轮交替顺序，
每轮 15 次，共各 60 次，不计编译和预热。测量时没有并发 GPU 测试。

| 模式 | 完整推理中位耗时 |
| --- | ---: |
| 关闭 Triton | 280.464 ms |
| 开启 Triton | 272.594 ms |

本次测量延迟降低 **2.806%**，加速 **1.029 倍**。收益较小，机器负载与输入会影响结果，
因此默认保持关闭。原始逐次延迟保存在
[`benchmarks/triton_a6000_2026-10-22.json`](benchmarks/triton_a6000_2026-10-22.json)。

额外比较了 3 组不同图像和固定噪声的完整物理动作：全部通过
`atol=0.002, rtol=0.002`；最大绝对差为 **0.004380**，平均绝对差为 **0.000574**。
该容差按 `abs(actual-reference) <= 0.002 + 0.002*abs(reference)` 判断，结果并非逐位一致。
这些检查验证数值接近程度和推理延迟，不包含机器人实机任务成功率评估。

## 两边代码一致性

两边起始 HEAD 均为 `b25b65d3d4f313e3bddec8d35d52d215400b3fce`。远端实现和测试后取回
本地，比较本次修改文件的完整 Git patch（包含新增文件）以及 SHA-256。
远端原有 `config/train/openpi.yaml` 的 `eval_enabled: true` 修改保留，本地原值不变；
这处训练配置不属于加速变更。
