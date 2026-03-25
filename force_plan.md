现在的架构是基于openpi的，我想要做一个修改。
过去的openpi不存在力的融合，我现在要在里面加入力的融合。

数据改动
原始采集的数据格式为：{
            "exterior_image_1_left",
            "exterior_image_2_left",
            "wrist_image_left",
            "gripper_image_left",
            "gripper_image_right",
            "joint_position",
            "gripper_position", 
            "actions",
}
通过forcevla_policy.py(policy决定了最终给到训推的数据)进行transfrom得到{

}

架构改动
vlm部分不做改动
wrist&&waibu + text --->vlm--->prefix_embs

action expert部分
1.力的处理
柔性夹爪的形变通过两个摄像头拍摄到的img(时序的)--->CBAM (Convolutional Block Attention Module)--->GRU--->project到应该的layer--->force_embs

将force_embs的后面拼接一个用 self.param 声明一个带有初始化的权重矩阵，并在前向传播时把它加(数值)到你的特征上。
类似这样：
self.force_modality_embed = self.param(
    "force_modality_embed",
    nn.initializers.normal(stddev=0.02), # 正态分布初始化
    (1, 1, self.expert_dim) # 形状: (Batch_size=1, Seq_len=1, Dim) 以支持广播
)
加上这个偏置参数后，即便 force_embs 和 prefix_embs 在数值上的分布有时候会很接近，AdaMoE 的路由器（也就是你代码里的 self.w_gating）在计算 jnp.dot 时，也能立刻识别出“带有力觉条形码”的 Token，从而把它们稳稳地导向专门处理力的专家。
这个的好处是保证力和moe的结合。

2.力通过moe进行融合(核心创新模块)
torch.cat[force_embs,prefix_embs]--->layernorm+self_attention+残差(参考pi0_force现有的融合方式)---> layernorm+moe(这里的架构是参考pi0_force，但是moe换成了src/openpi/models/moe.py这里的moe新架构adamoe，具体怎么组织adamoe这个新的moe的架构参考src/openpi/models/gemma_adamoe.py
src/openpi/models/gemma.py这两个的差异，gemma_adamoe是用了这个新的adamoe的，只学习怎么组合的block，而不是真的修改自己的gemma.py)--->linear(参考pi0_force)--->fvl_features

3.传入到action expert
fvl_features--->action expert

4.action expert 的外面分支挂件

我们保留了 Action Expert 原始的 8 维输出绝对不碰，而是在它的旁边，平行拉出了一个小分支（Compliance Head）。最后在模型内部把它们乘起来，对外依然只交出 8 维数据。

具体在 src/openpi/models/pi0_force.py 里的逻辑变成了这样：

Python
# 1. 主分支：原封不动的 Action Expert（输出原始预测的 8 维运动趋势）
base_delta_action = self.action_expert(fvl_features) # 形状: (B, L, 8)
# 2. 旁路分支：新增一个独立的 Compliance Head（专管柔顺度）
c_logits = self.compliance_head(fvl_features) # 一个极小的全连接层
c_factor = nn.sigmoid(c_logits) # 压到 0~1 之间，形状: (B, L, 1)
# 3. 内部调制（核心！）
# 用柔顺度系数去缩放原始预测的动作
modulated_delta_action = base_delta_action * c_factor 
# 4. 返回给 OpenPI 框架
# 外层框架根本不知道 c 的存在，它收到的依然是规规矩矩的 8 维动作
return modulated_delta_action

为了放置这个c乱动失去物理意义
引入动作正则化惩罚 (Action Regularization)我们不需要真实的力学标签，只需要给 $\Delta P_{predict}$ 加一个 $L_2$ 正则化惩罚。在你的 Loss 函数计算里，加上这一项：$$Loss_{total} = Loss_{action} + \lambda ||\Delta P_{predict}||_2^2$$物理意义：这逼迫网络输出的 $\Delta P_{predict}$ 必须是“合理的常规动作幅度”，绝对不能是个天文数字。这样一来，当网络需要输出极小的最终动作时，它就被迫只能通过降低 $c$ 值来实现。这就完美地让 $c$ 承担了“柔顺度”的物理责任。

5.大小脑异步推理架构：
数据采集与训练阶段
在采集专家示教数据时，你的硬件频率设定是完全对的，但有一个极其关键的隐藏条件——动作标签的频率。
外部传感器 (慢)： 手腕相机、外部全局相机的采集频率锁定在 20Hz（每 50ms 产出一帧）。
内部传感器和本体状态与动作标签 (快)： 拍摄柔性夹爪内部形变的相机采集频率拉高到 60Hz（每 16.7ms 产出一帧）。 记录 Franka 机械臂的 joint_position、gripper_position 以及你下发的 actions，必须也是 60Hz。
训练时的数据拼接
在构建送入网络的 Batch 时，你要以 60Hz 的快时钟为主轴。假设在一个 50ms 的时间窗口里，你有 3 个快时钟节点（$T_1, T_2, T_3$）：在 $T_1$ 时刻：你有最新的全局图像 $V_1$ 和最新的夹爪图像 $G_1$。拼接起来预测动作 $A_1$。在 $T_2$ 时刻：夹爪图像更新到了 $G_2$。但全局图像还没更新！ 此时你直接复制（缓存） $V_1$，把 $(V_1, G_2)$ 拼接起来，预测动作 $A_2$。在 $T_3$ 时刻：夹爪图像更新到了 $G_3$，全局图像依然用 $V_1$。拼接 $(V_1, G_3)$ 预测动作 $A_3$。
推理的时候：
因为输入是vl,所以和上述的训练的情况是一样的，快手指匹配慢vla相机，来完成数据的拼接。在 $T_2$ 和 $T_3$ 时刻，如果你缓存的是原始全局图像 $V_1$，并在每次快时钟节点都将 $(V_1, G_2)$ 一起送入一个庞大的网络进行完整的前向传播，你的推理硬件极大概率撑不住 60Hz 的运行频率。优化建议：不能把 $V_1$ 重新过一遍 ViT 和 LLM。你需要把 $V_1$ 生成的 KV Cache 锁住并复用。此时，你只需要将最新的快时钟夹爪/本体状态 $G_2, G_3$（转化为极少量的 State Tokens）作为新的输入送进去，利用现成的图像 KV Cache 来自回归地预测动作 $A_2, A_3$。你缓存并输入给下游动作预测网络的是 $F_1$，而不是重新对 $V_1$ 跑一遍vlm。


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
        ....
    ),
    这个新的config来完成我的训练和推理过程，并在上述的代码里更新我的架构要求


需要做的实验
1.特殊的需要力的任务并且消融
2.做个可视化，证明 AdaMoE 自动把任务分成了几个专家（重要）
视觉”和“力”分成了几个专家，最直观的做法就是画一张随时间变化的专家权重热力图。
拦截 gating_scores_softmax 并画图
数据拦截：在你写的 AdaMoEFeedForward.gating_topk 方法里，gating_scores_softmax 记录了每个 Token 分配给 4 个专家的概率。在执行推理（Inference/Rollout）时，把每一步（Step）里带有 force_modality_embed 标签的那个 Token 的分配概率存进一个列表。
代码绘制：跑完一次完整的抓取任务后，你会得到一个尺寸为 (总步数, 专家数量) 的 NumPy 数组。用 Python 的 seaborn 或 matplotlib 画出来：
X轴：时间步（比如从 0 到 100 步）。
Y轴：专家 ID（Expert 0 到 Expert 3）。
颜色深浅：代表该专家被分配的权重。
<!-- 预期效果：在图表的前半段（手臂在空中移动），Expert 0/1 亮起；在第 50 步夹爪接触物体、形变突增的瞬间，图表上的高亮区域瞬间切换到 Expert 2/3。这张图放进论文或汇报里，就是绝杀。 -->
3.执行 Modality Dropout：在构建 Batch 数据送入网络前，设置一个概率（例如 $p = 0.2$）。视觉致盲：以这个概率，将外部主视角相机的图像张量（Tensor）替换为全 0（黑图）或高斯噪声。做图像的dropout,防止视觉的特征太强大而忽略的力的特征，且需要消融（重要）