现在的架构是基于openpi的，我想要做一个修改。
过去的openpi不存在力的融合，我现在要在里面加入力的融合。

vlm部分不做改动
wrist&&waibu + text --->vlm--->prefix_embs

action expert部分
1.力的处理
柔性夹爪的形变通过两个摄像头拍摄到的img--->CBAM (Convolutional Block Attention Module)--->project到应该的layer--->force_embs

2.力通过moe进行融合(核心创新模块)
torch.cat[force_embs,prefix_embs]--->layernorm+self_attention+残差(参考pi0_force现有的融合方式)---> layernorm+moe(这里的架构是参考pi0_force，但是moe换成了src/openpi/models/moe.py这里的moe新架构)--->linear(参考pi0_force)--->fvl_features

3.传入到action expert
fvl_features--->action expert
