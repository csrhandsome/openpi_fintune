现在的架构是基于openpi的，我想要做一个修改。
过去的openpi不存在力的融合，我现在要在里面加入力的融合。

数据改动
原始采集的数据格式为：{

}
通过forcevla_policy.py(policy决定了最终给到训推的数据)进行transfrom得到{

}

架构改动
vlm部分不做改动
wrist&&waibu + text --->vlm--->prefix_embs

action expert部分
1.力的处理
柔性夹爪的形变通过两个摄像头拍摄到的img(时序的)--->CBAM (Convolutional Block Attention Module)--->GRU--->project到应该的layer--->force_embs

2.力通过moe进行融合(核心创新模块)
torch.cat[force_embs,prefix_embs]--->layernorm+self_attention+残差(参考pi0_force现有的融合方式)---> layernorm+moe(这里的架构是参考pi0_force，但是moe换成了src/openpi/models/moe.py这里的moe新架构adamoe，具体怎么组织adamoe这个新的moe的架构参考src/openpi/models/gemma_adamoe.py
src/openpi/models/gemma.py这两个的差异，gemma_adamoe是用了这个新的adamoe的，只学习怎么组合的block，而不是真的修改自己的gemma.py)--->linear(参考pi0_force)--->fvl_features

3.传入到action expert
fvl_features--->action expert

目前状态
1.将所有的相关的代码都放到这个目录下面了，要求先查看对应的这些代码。可以注意去查看现在的这些代码和原版代码的对应关系。
src/openpi/models/gemma_adamoe.py
src/openpi/models/image_process_model.py
src/openpi/models/pi0_force.py
src/openpi/models/moe.py
src/openpi/models/limoe_simple.py
src/openpi/training/config.py
src/openpi/training/weight_loaders.py

从config对应的出发，通过
    TrainConfig(
        name="forcevla_lora",
        model=pi0_force.Pi0_GuidanceConfig(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotForcevlaDataConfig(
            repo_id="flexiv_peel_cucumber_inputForce",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.Pi0GuidanceWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=50_000,
        freeze_filter=pi0_force.Pi0_GuidanceConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=4,
    ),
    这个新的config来完成我的训练和推理过程，并在上述的代码里更新我的架构要求


需要做的实验
1.特殊的需要力的任务并且消融
2.做个可视化，证明 AdaMoE 自动把任务分成了几个专家（重要）
3.做力的dropout,防止视觉的特征太强大而忽略的力的特征，且需要消融（重要）