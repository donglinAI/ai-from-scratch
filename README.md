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
| 02-rotary-position-embedding | ✓ | [小红书分享](https://www.xiaohongshu.com/explore/6ad1ba3a00000000150052f8?xsec_token=ABQLRD8o_3HJsGCDwgRpJa1bf-p-8FKj3DAtyvIdiEeIw=&xsec_source=pc_user) | [小红书分享](https://www.xiaohongshu.com/explore/6ad31eb3000000001a0211d2?xsec_token=ABKTNiA1JoS16skb7UebVOv_Rkrq_uTDf72VRh1ho_Ej4=&xsec_source=pc_user) |
| 03-self-attention | ✓ | [](#) | [](#) |
| rms-norm | ☐ | [](#) | [](#) |
| swiglu | ☐ | [](#) | [](#) |
| optimizer-adamw | ☐ | [](#) | [](#) |

## 🚧 路线图（持续更新中）
> 按"由底层到工程、由单卡到多卡、由训练到推理"的顺序整理，覆盖大模型训练 / 推理 / AI Infra 的一些基础知识。

### 🧱 一、基础组件层

- ✓ `backpropagation-from-scratch` —— 纯 Python 手动实现反向传播
- ✓ `rotary-position-embedding` —— RoPE 旋转位置编码
- ✓ `self-attention` —— 多头自注意力 / 分组查询注意力（MHA & GQA）
- ☐ `rms-norm` —— RMSNorm 归一化
- ☐ `swiglu` —— SwiGLU 激活函数
- ☐ `parameter-init` —— 权重初始化策略
- ☐ `tokenizer` —— 分词器（BPE / SentencePiece）
- ☐ `optimizer-adamw` —— AdamW 优化器
- ☐ `lr-scheduler` —— 学习率调度（Warmup / Cosine / 多项式）
- ☐ `gradient-clip` —— 梯度裁剪
- ☐ `mixed-precision` —— 混合精度训练（AMP / BF16 / FP8）

### 🏗️ 二、模型手写层

- ☐ `transformer` —— 完整 Transformer（Encoder-Decoder）
- ☐ `llama-from-scratch` —— Llama 手写实现（组装 GQA / RoPE / RMSNorm / SwiGLU）

### ⚙️ 三、训练优化层

- ☐ `memory-estimation` —— 显存估算（参数 / 激活 / 梯度 / KV Cache 的占用）
- ☐ `gradient-accumulation` —— 梯度累积
- ☐ `activation-checkpointing` —— 激活重计算（省显存）
- ☐ `llama-finetune` —— Llama 微调
- ☐ `lora-finetune` —— LoRA / QLoRA 参数高效微调
- ☐ `train-stability` —— 训练稳定性调试（loss spike / 梯度爆炸排查）

###  四、强化学习 / 对齐层（RLHF & Post-Training）

- ☐ `rlhf-overview`  —— RLHF 整体流程（SFT → RM → RL 三阶段）
- ☐ `reward-model`  —— 奖励模型（RM）训练
- ☐ `ppo`  —— PPO 策略优化算法
- ☐ `grpo`  —— GRPO 分组相对策略优化（DeepSeek 风格）
- ☐ `dpo`  —— DPO 直接偏好优化（无需奖励模型）
- ☐ `rollout-data`  —— Rollout 采样与偏好数据管理

### 🌐 五、分布式并行层

- ☐ `multi-gpu` —— PyTorch 多卡并行基础
- ☐ `ddp` —— DDP 数据并行算法
- ☐ `comm-cost` —— 通信开销分析（all-reduce 通信量，DDP vs ZeRO 差异）
- ☐ `zero-1-2-3` —— ZeRO 分片优化（1 / 2 / 3 阶段）
- ☐ `tensor-parallel` —— 张量并行
- ☐ `pipeline-parallel` —— 流水线并行

### ⚡ 六、推理部署层

- ☐ `kv-cache` —— KV Cache 增量推理原理与实现
- ☐ `attention-optimization` —— 高效注意力（Flash-Attention / 窗口注意力）
- ☐ `inference-engine` —— 推理引擎原理（PagedAttention / vLLM）
- ☐ `speculative-decoding` —— 投机解码（加速生成）
- ☐ `quantization` —— 量化推理（GPTQ / AWQ / FP8）

### 🚀 七、端到端实战层

- ☐ `llama-parallel-train` —— Llama 多卡并行训练完整流程
- ☐ `llama-deploy` —— Llama 推理部署与性能优化
