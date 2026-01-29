# DROID pi05 微调说明（必读）

- 第一次查看/修改前：先用 rg 搜索关键词（pi05_full_droid_finetune、pi05_droid_finetune、RLDSDroidDataConfig、LeRobotDROIDDataConfig、action_space、DeltaActions），确认当前配置与数据语义。
- 动作语义由 data config 决定，不是模型架构决定。
- pi05_full_droid_finetune：RLDS + JOINT_POSITION，训练时把绝对关节位置转为 delta，输出再还原为关节位置。
- pi05_droid_finetune：LeRobot 格式，默认假设 joint velocity，不做 delta 变换，输出为速度。

# 逻辑代码位置

- RLDSDroidDataConfig 与 JOINT_POSITION 的 delta/absolute 逻辑：src/openpi/training/config.py:358、src/openpi/training/config.py:403
- LeRobotDROIDDataConfig 的速度假设：src/openpi/training/config.py:426、src/openpi/training/config.py:450
- 具体训练配置：pi05_full_droid_finetune 在 src/openpi/training/config.py:861；pi05_droid_finetune 在 src/openpi/training/config.py:895
- action_space 枚举：src/openpi/training/droid_rlds_dataset.py:21
- DeltaActions/AbsoluteActions 实现：src/openpi/transforms.py:204、src/openpi/transforms.py:226

# 微调方案需要关注的 key

pi05_full_droid_finetune（RLDS/全量）
- data.rlds_data_dir（必须）
- data.action_space（JOINT_POSITION 或 JOINT_VELOCITY）
- data.assets.assets_dir / data.assets.asset_id（建议沿用 pi05_base 资产）
- data.datasets（版本/权重/过滤索引）
- model.action_dim / model.action_horizon
- weight_loader
- num_workers=0（RLDS 必须）

pi05_droid_finetune（LeRobot/小数据）
- data.repo_id（你的 LeRobot 数据集）
- data.base_config.prompt_from_task
- data.assets.assets_dir / data.assets.asset_id（建议沿用 pi05_droid 资产）
- model.action_dim / model.action_horizon
- weight_loader
