# AI From Scratch

## 这个项目是什么

**本项目的核心目的：使用代码从零实现大模型相关的核心算法。**

用纯 PyTorch 手写大模型的底层基础组件（反向传播、RoPE、注意力、归一化、优化器、分布式并行、强化学习对齐等），不依赖高层框架的黑盒调用，把"算法是怎么从数学落到代码"一步步拆开实现。每个模块都带单元测试，并与 HuggingFace 官方实现做数值对齐校验，尽量保证实现正确、能对齐。

整个过程是个人技术栈重塑的实战记录，也配有讲解视频（理论 + 代码），欢迎一起交流、互相指正。

## 🎬 配套讲解视频
> 每个模块录两个视频（理论推导 + 代码讲解），欢迎多提意见：

| 模块 | 状态 | 理论讲解 | 代码讲解 |
|:-----|:---:|:---|:---|
| 01-backpropagation-from-scratch | ✓ | [](#) | [](#) |
| 02-rotary-position-embedding | ✓ | [旋转位置编码-理论篇](https://www.xiaohongshu.com/explore/6ad1ba3a00000000150052f8?xsec_token=ABQLRD8o_3HJsGCDwgRpJa1bf-p-8FKj3DAtyvIdiEeIw=&xsec_source=pc_user) | [旋转位置编码-代码篇](https://www.xiaohongshu.com/explore/6ad31eb3000000001a0211d2?xsec_token=ABKTNiA1JoS16skb7UebVOv_Rkrq_uTDf72VRh1ho_Ej4=&xsec_source=pc_user) |
| 03-self-attention | ✓ | [自注意力机制-理论篇](https://www.xiaohongshu.com/explore/6ad70acf000000002f03237d?xsec_token=ABj7eZjkQn7TJR4pfqLdYHEmtDECd8NPpmPz8YVDwI5cs=&xsec_source=pc_user) | [自注意力机制-代码篇](https://www.xiaohongshu.com/explore/6ad89540000000002c03b9e5?xsec_token=ABLnasY4rP1yhaEgpoGOe3pPNWXa2oSV_5fbys4lX9pGo=&xsec_source=pc_user) |
| 04-rms-norm | ✓ | [均方根层归一化-理论篇](https://www.xiaohongshu.com/explore/6ae176b400000000340245ec?xsec_token=ABEUFAX6mC_j6-TeXgzNjC4bMFgzzrJJIDSGeHB6JS3mo=&xsec_source=pc_user) | [均方根层归一化-代码篇](https://www.xiaohongshu.com/explore/6af3d658000000002b010a94?xsec_token=ABls6L9vohmIqvtO3HjjVjAVn0065pAzCq-sFRUrZ95G0=&xsec_source=pc_user) |
| 05-relu | ✓ | [ReLU激活函数-理论篇](https://www.xiaohongshu.com/explore/6ae5cd66000000001200b47b?xsec_token=ABZDUmuy1o3zs5tBuIW9VNi-2yYAlEoQTd2zymzTPHmpU=&xsec_source=pc_user) | [](#) |
| 06-swiglu | ✓ | [SwiGLU激活函数-理论篇](https://www.xiaohongshu.com/explore/6ae730070000000006021a0b?xsec_token=YBHEE_PYJMEN8q-eGa1fBAl3RL1S1MyRByzo4coXDd14w=&xsec_source=pc_creatormng) | [](#) |
| 07-optimizer-adamw | ✓ | [AdamW优化器-理论篇](https://www.xiaohongshu.com/explore/6ae81e580000000019002e2c?xsec_token=YBodqmf6y_frpJyAv4SP0qiiCCW5_hd38xbPaCI5fA4Qs=&xsec_source=pc_creatormng) | [](#) |
| 08-lr-scheduler | ☐ | [](#) | [](#) |
| 09-transformer | ☐ | [](#) | [](#) |
