# torch.compile 部署开关与远端验证

在 `config/server/openpi.yaml` 或 `config/server/openpi_rlt.yaml` 中配置：

```yaml
device: cuda
use_torch_compile: true
use_triton: false
```

两个加速开关相互独立，默认均为 `false`，可以同时开启。命令行覆盖方式：

```bash
python scripts/server.py --config config/server/openpi.yaml --use-torch-compile True
```

部署加载权重、移动到 CUDA 并设置 eval 后，使用 Inductor 默认模式、
`fullgraph=True`、`dynamic=False` 编译 `run_suffix`（单个去噪步骤），Euler 循环
复用后缀图。视觉编码、语言前缀、预处理、采样随机噪声与动作后处理保持原路径。
VLA、RLT 与 force 服务均传递配置，共享前缀的 RL-token 路径也调用相同的编译方法。
actor、critic 和 readout 不在此次编译范围内。

编译选项还包含 `emulate_precision_casts=True`（保留 BF16 中间转换）和
`force_same_precision=True`（约束 TF32 精度选择）。在真实 checkpoint 上同时编译
视觉/语言前缀时曾出现更大的动作偏差，因此最终仅编译重复执行的去噪步骤。
这些设置减小了差异，但不保证与原路径逐位一致，也不保证严格动作容差通过。

开启要求 CUDA PyTorch eval 模型；CPU/JAX 配置会明确报错。只替换模型实例的推理方法，
不包装子模块，不改变 state_dict 键或训练配置。默认关闭不影响已有 GELU/RoPE 等小算子的
编译设置。首次请求按输入形状延迟编译，可能耗时数分钟；新形状、dtype、模式也可能重新
编译，部署应在实际请求形状下预热。编译错误直接暴露，不会静默退回 eager。
当前默认模式不启用 CUDA Graph，避免跨前缀缓存和多次去噪调用的张量生命周期问题。

Gemma 的缓存类型判断改为 `isinstance(kv_cache, (list, tuple))`：该写法与原逻辑等价，
避免 PyTorch 2.7 TorchDynamo 无法追踪 `list | tuple` 的 `UnionType`。
编译选项定义参见 [PyTorch 2.7 官方文档](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html)。

## 复现

```bash
JAX_PLATFORMS=cpu SWANLAB_MODE=disabled .venv/bin/python -m pytest \
  src/openpi/tests/models_pytorch/compile_inference_test.py \
  src/openpi/tests/models_pytorch/triton_inference_test.py \
  src/openpi/tests/models_pytorch/pi0_pytorch_test.py \
  src/openpi/tests/serving \
  src/openpi/tests/rlt/force_rlt/yaml_stage_test.py -q

JAX_PLATFORMS=cpu SWANLAB_MODE=disabled .venv/bin/python scripts/benchmark_compile.py \
  --warmup 5 --repeats 15 --rounds 4 \
  --output /tmp/openpi-compile-benchmark.json
```

基准加载同一个真实 checkpoint，使用合成 Franka 双相机图像、状态和固定噪声，
分别比较原路径、Triton、torch.compile、torch.compile + Triton。计时覆盖 `Policy.infer`
的输入与输出变换，并在前后同步 CUDA。各模式预热后分四轮轮换顺序测量，各 60 次。
首次调用（包含可能的编译）单独记录，不计入稳态延迟；远端存在编译缓存，不是
全新环境的冷启动耗时。额外使用三组图像和噪声，以 `atol=0.002, rtol=0.002`
检查最终物理动作，并记录失败元素数、各动作维度最大误差与均值。
若严格动作检查失败，基准仍完成计时和 JSON 保存，最后以非零状态退出，
不会将精度失败记为通过。检查不包含机器人任务成功率；该阈值也不是机器人控制验收标准。


## 2026-10-22 远端结果

远端：`mirnagroup:/home/mirnagroup/code/openpi_fintune`，NVIDIA RTX A6000，
PyTorch 2.7.1+cu128、Triton 3.3.1。真实 checkpoint：
`checkpoints/openpi/joint_7_6_audio_qwen_asr/30000`，Pi0 BF16、LoRA rank 48、
动作维度 32、预测长度 50、Euler 10 步。GPU 基准与 CUDA 回归顺序运行。

最终回归结果：**69 passed**，包含 Pi0/Pi0.5、LoRA、共享前缀、Triton 组合、
编译开关的配置与覆盖、VLA/RLT/force 传递以及现有 serving 测试。

| 模式 | 端到端中位耗时 | 相对原路径加速 |
| --- | ---: | ---: |
| 原路径 | 292.978 ms | 1.000× |
| 仅 Triton | 306.458 ms | 0.956× |
| 仅 torch.compile | 136.209 ms | 2.151× |
| torch.compile + Triton | 136.312 ms | 2.149× |

仅 torch.compile 的延迟降低 **53.51%**。
本轮组合 Triton 没有进一步提速；仅 Triton 比原路径稍慢，与此前独立 Triton 基准的
小幅收益不同，说明这部分收益受测量波动和运行环境影响。各模式逐次延迟、首次调用
耗时和配置见 [原始 JSON](benchmarks/torch_compile_a6000_2026-10-22.json)。

### 精度结果：严格检查未通过

所有输出有限，三组物理动作总共 1200 个元素。保持之前的严格判定：
`abs(actual-reference) <= 0.002 + 0.002*abs(reference)`，没有放宽阈值。

| 模式 | 严格检查 | 超差元素 | 最大绝对差 | 平均绝对差 |
| --- | --- | ---: | ---: | ---: |
| 仅 Triton | 通过 | 0/1200 | 0.004380 | 0.000574 |
| 仅 torch.compile | 未通过 | 30/1200 | 0.004570 | 0.000455 |
| torch.compile + Triton | 未通过 | 36/1200 | 0.005656 | 0.000878 |

编译路径有明确的速度收益，但当前真实 checkpoint 未满足该严格动作容差。
因此 `use_torch_compile` 默认保持关闭；启用前需要按实际任务评估动作误差及实机表现。
上述 **69 passed** 是回归套件结果，不代表真实 checkpoint 的严格精度检查通过；
本次基准保存结果后以状态码 1 退出，回归套件以状态码 0 退出。

## 同步与差异核对

两边起始 HEAD 均为 `50cf3a8c0470c4c82bab3abffd064f326ed39a15`。
实现和最终测试均先在远端完成，再取回本地代码、测试、文档与原始数据。
比较本次所有文件的 SHA-256，以及相对同一 HEAD 的完整 Git patch（包含新增文件）。
远端原有 `config/train/openpi.yaml` 中 `eval_enabled: true` 的修改保留，不同步至本地。
不创建 Git commit，也不修改依赖版本。
