# VLA 验证集与动作 MSE 检查

当前普通 VLA 训练已经接入整条轨迹切分与去噪后动作 MSE。检查范围是 `config/train/openpi.yaml`、数据加载器、Pi0 Euler 采样器以及训练中的验证调用。

训练 YAML 中配置 `eval_enabled: false`、`val_ratio: 0.05`、`split_seed: 42`、`eval_interval: 500`、`eval_num_steps: 10`。将 `eval_enabled` 改为 `true` 才启用下面的验证切分与评估；缺少开关的旧配置也默认关闭。配置解析、命令行覆盖和 checkpoint 配置快照均保留这些字段。`val_ratio: 0` 也关闭验证。

关闭验证时直接调用原来的 `create_data_loader(config, shuffle=True, batch_size=...)`，全部数据参与训练，不读取 episode 切分索引、不创建验证 loader。预览图继续单独创建原 loader，保留原来的 worker、shuffle、batch 和随机数使用路径；训练期间不调用验证采样器。

此仓库使用 LeRobot，切分单位为 episode。按 episode 编号排序后，用 `RandomState(split_seed)` 打乱，取前 `max(1, int(N * val_ratio))` 条验证。训练和验证包含各自 episode 的所有帧。它不会按 Parquet/MP4 物理文件切分，因为 LeRobot v3 可以把多条 episode 合并到一个文件。只有一条 episode 时，开启验证会报错。

2026-10-19 读取当前 YAML 指向的数据集元数据，得到 121 条 episode、52026 帧。启用验证后，验证集为 episode `[44, 47, 4, 55, 26, 64]`，共 6 条、2446 帧；训练集为 115 条、49580 帧。两边帧索引无重叠，验证帧占比约 4.70%。默认关闭时，121 条全部用于训练。

每到验证间隔，模型切到 eval 模式并关闭梯度，从标准高斯噪声出发，Euler 积分 10 步，由 t=1 到 t=0，动作块长度跟随 `model.action_horizon`，当前为 50。采样预处理明确使用 `train=False`，关闭图像增强。预测直接和同一套数据变换及 `norm_stats` 产生的真值比较，没有反归一化。

`eval_loss` 是有效动作维度上平方误差的总和除以元素总数。当前 joint 配置使用前 8 维：7 维关节与 1 维夹爪，排除模型补齐到 32 维的部分，另记 `eval_l2_joint` 和 `eval_l2_gripper`。EE/EE-delta 按其实际 XYZ、RPY、夹爪布局记录分量。这里没有参考项目的 rot6d 或 progress 通道，因此不会固定截取 21 维。

验证不会使用训练的速度场 loss、action indicator 或额外对齐 loss。它保留末尾不足一个 batch 的样本，并按实际元素数加权，因此尾 batch 不会被赋予与完整 batch 相同的权重。episode 末端补出的未来动作时间步也参与 MSE，没有额外时间 mask。多卡代码对验证索引按 rank 步进分片，汇总平方误差与元素数；本次没有执行真实多卡训练。

本次补充的回归检查覆盖部分 batch 的权重、异常时恢复模型模式、10 步 Euler 的积分终点、真实 LeRobot MP4 episode 的加载与共享归一化、关闭验证，以及两次优化器更新间的完整验证调用。训练 smoke test 使用小 Gemma/SigLIP 和合成数据；没有运行完整 2B 模型的 GPU 训练。

下方命令结果为 **67 passed, 1 deselected**；排除项是需要人工准备真实数据的 manual 测试。相关代码的 Ruff 检查、测试文件格式检查和 `git diff --check` 通过。

增加开关后，受影响的配置、数据加载和训练回归检查为 **50 passed, 1 deselected**，覆盖开关类型校验、旧 YAML 缺字段、CLI 开关、快照恢复、关闭时跳过切分、独立预览及开启时正常验证。

另从 Git HEAD `8cb804b` 直接读取原 `vla.py` 和 `data_loader.py`，在相同 seed、配置、输入及小 Gemma/SigLIP 下，与关闭验证的新版本各运行 3 次优化器更新并跨过 epoch 边界。逐项精确比较得到：3 步 loss 相同、64 个权重 tensor 相同、优化器状态相同、Torch/NumPy 随机数状态相同；两边均构建一次训练数据和一次预览数据。此对照使用 CPU 合成数据，不表示完整 GPU/DDP 训练的逐位结果已经验证。

默认配置的动作类型、动作维度、帧率、LoRA rank 及验证参数由测试独立读取 YAML 获取预期值。格式与算法测试中的 EE/joint、episode 长度、固定采样步数等仍是显式测试样例，不依赖默认实验。默认数据校验测试也按所选 YAML 生成相应数据 schema 和 fps。

真实训练仍需先提供当前数据集的 `norm_stats.json`；检查时默认的 `data/assets` 目录尚不存在。测试用已知统计量验证了归一化路径。

复现检查时，本环境需要禁用与项目无关的 ROS pytest 插件，并将 Hugging Face 数据集缓存放在可写目录：

```bash
JAX_PLATFORMS=cpu \
HF_DATASETS_CACHE=/tmp/openpi-validation-hf-cache \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 \
.venv/bin/python -m pytest -q -m 'not manual' \
  src/openpi/tests/training/action_eval_test.py \
  src/openpi/tests/training/data_loader_test.py \
  src/openpi/tests/training/config_test.py \
  src/openpi/tests/training/lerobot_v3_test.py \
  src/openpi/tests/training/train_test.py::test_training_evaluates_denoised_actions_and_continues \
  src/openpi/tests/training/train_test.py::test_default_ros_config_and_dataset_validation
```
