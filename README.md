# AI From Scratch

**从数学原理到可运行代码，手写跨模态 AI 核心算法。**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/donglinAI/ai-from-scratch?style=social)](https://github.com/donglinAI/ai-from-scratch/stargazers)

一个面向 **AI 基础算法与核心模型** 的手写复现项目。项目使用 Python 和 PyTorch，从自动微分、网络组件、优化算法等基础能力出发，逐步扩展到大语言模型、计算机视觉、多模态、图像生成与 Diffusion 等方向。

项目的目标不是复现某一个模型，而是建立一套**跨模态、跨任务、可持续扩展的 AI 手写算法体系**：既拆解 Llama、ViT 等具体模型，也实现它们背后可复用的基础算法，重点讲清楚三个问题：

1. **数学公式如何转化为代码**：保留关键计算步骤、张量形状和梯度传播过程。
2. **模型组件如何协同工作**：从 RoPE、Attention、RMSNorm 等基础组件，走到完整模型训练或推理链路。
3. **实现结果如何验证**：通过内置断言、参考实现对照、数值检查和可视化实验检验正确性。

本项目以学习和复现为目标，不追求替代生产级训练或推理框架。代码尽量保持单文件、少封装、可运行，适合逐行阅读和二次修改。

## 项目特点

- **跨领域算法版图**：覆盖深度学习基础、大语言模型、计算机视觉，并持续扩展多模态、图像生成与 Diffusion。
- **从零实现核心算法**：不局限于调用现成模型，重点手写自动微分、注意力、归一化、优化器、损失函数及生成模型关键组件。
- **贯通完整模型链路**：既提供可独立阅读的最小算法实现，也逐步拆解完整模型的数据流、训练与推理过程。
- **重视数值正确性**：部分模块与 PyTorch 或 Hugging Face 参考实现进行对照，并提供可直接运行的自检脚本。
- **兼顾理论与直觉**：结合中文笔记、公式推导、张量形状注释和可视化结果，降低底层原理的理解成本。
- **配套视频讲解**：以“理论推导 + 代码实现”的方式持续更新，方便按模块系统学习。

## 内容版图

下表区分当前代码与后续方向，避免将规划内容误认为已经实现：

| 方向 | 当前进展 | 已覆盖或计划实现的内容 |
|:---|:---|:---|
| AI 数学与深度学习基础 | 已覆盖，持续补充 | 自动微分、反向传播、激活函数、归一化、损失函数、优化器、学习率调度 |
| 大语言模型 | 已覆盖，持续扩展 | BPE、RoPE、MHA / GQA、SwiGLU、KV Cache、Llama 3.1 推理，以及后续训练与对齐算法 |
| 计算机视觉 | 已起步 | Vision Transformer；后续扩展 CNN、ResNet、目标检测、图像分割等基础算法 |
| 多模态 | 规划中 | 跨模态注意力、对比学习、CLIP、视觉语言模型，以及图文检索与理解 |
| 图像生成与 Diffusion | 规划中 | VAE、GAN、U-Net、DDPM、DDIM、Latent Diffusion、Stable Diffusion 等核心原理与最小复现 |
| 训练与推理系统 | 规划中 | 分布式训练、混合精度、量化、推理加速，以及 CUDA / Triton 算子 |

## 模块导航

| 模块 | 主题 | 当前内容 |
|:---|:---|:---|
| [01](./01-backpropagation-from-scratch) | 反向传播与自动微分 | 手写标量计算图、拓扑排序与反向传播，并用 MLP 求解 XOR 问题；提供 PyTorch 对照实现 |
| [02](./02-rotary-position-embedding) | 旋转位置编码 RoPE | 标准 RoPE、Llama 3.1 三段式频率缩放、频率曲线可视化与中文推导笔记 |
| [03](./03-self-attention) | 自注意力机制 | MHA / GQA、RoPE、因果掩码与缩放点积注意力；提供 Hugging Face 数值对照 |
| [04](./04-rms-norm) | RMSNorm | RMSNorm / LayerNorm、PreNorm / PostNorm 对比实验，以及 RMSNorm 解析梯度推导 |
| [05](./05-relu) | ReLU | 自定义 `autograd.Function` 前向与反向传播、激活曲线和死亡神经元实验 |
| [06](./06-swiglu) | SwiGLU | 门控前馈网络、SiLU 性质、激活与梯度曲线，以及与 Llama MLP 的数值对照 |
| [07](./07-adamw) | AdamW | 一阶矩、二阶矩、偏差修正与解耦权重衰减；与 `torch.optim.AdamW` 对照 |
| [08](./08-lr-scheduler) | 学习率调度 | Linear Warmup、Cosine Decay、多参数组与调度器状态保存 / 恢复 |
| [09](./09-cross-entropy-loss) | 交叉熵与 Focal Loss | 类别权重、`ignore_index`、Label Smoothing、Focal Loss，以及信息论可视化 |
| [10](./10-llama-from-scratch) | Llama 3.1 推理 | GQA、RoPE Scaling、RMSNorm、SwiGLU、KV Cache、Safetensors 权重加载与文本生成 |
| [11](./11-vit-from-scratch) | Vision Transformer | Patch Embedding、CLS Token、位置编码、Pre-LN Transformer Block 与分类头 |
| [12](./12-environment) | 开发环境 | AutoDL 初始化脚本，以及 Git / GitHub 配置说明 |
| [13](./13-bpe) | BPE 分词器 | 字节级 BPE 训练、编码、解码、词表保存与加载 |
| [14](./14-kv-cache) | KV Cache | 静态缓存、Prefill / Decode、因果掩码，以及增量解码与全量重算的一致性检查 |

## 推荐学习路径与扩展路线

1. **AI 基础**：`01 → 05 → 09`，理解计算图、梯度、激活函数与损失函数。
2. **Transformer 核心组件**：`02 → 03 → 04 → 06`，掌握语言、视觉与多模态模型共同使用的基础模块。
3. **训练算法**：`07 → 08`，理解参数更新与学习率变化如何影响训练过程。
4. **大语言模型**：`13 → 14 → 10`，串联分词、缓存、模型结构和文本生成流程。
5. **视觉与多模态**：从 `11` 的 ViT 开始，后续扩展到视觉骨干网络、跨模态对齐和视觉语言模型。
6. **图像生成与 Diffusion**：从 Autoencoder、VAE 和 U-Net 等基础组件出发，逐步进入 DDPM、DDIM 与 Latent Diffusion 的完整生成流程。

可以按上述路线系统学习，也可以从一个具体模型出发，沿其计算流程回到对应的基础算法。例如，从 [Llama 3.1 单文件实现](./10-llama-from-scratch/llama31_inference.py) 入手，再回溯 RoPE、Attention、RMSNorm、SwiGLU 与 KV Cache。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/donglinAI/ai-from-scratch.git
cd ai-from-scratch
```

### 2. 准备环境

建议使用 **Python 3.10 或更高版本**。基础模块主要依赖 PyTorch，可视化脚本额外使用 NumPy 和 Matplotlib。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy matplotlib
```

如需 GPU，请根据 CUDA 版本选择对应的 [PyTorch 安装命令](https://pytorch.org/get-started/locally/)。

Llama 推理及部分参考实现对齐实验还需要以下依赖：

```bash
pip install transformers safetensors modelscope
```

### 3. 运行模块示例

大多数脚本都可以直接执行。包含本地导入的脚本，建议先进入对应目录再运行。

```bash
# RoPE：标准实现与 Llama 3.1 频率缩放
cd 02-rotary-position-embedding
python rope.py
python rope_llama3.py

# MHA / GQA 自注意力
cd ../03-self-attention
python attention.py

# BPE：训练、编码、解码及保存 / 加载自检
cd ../13-bpe
python bpe.py

# KV Cache：增量解码与全量重算一致性检查
cd ../14-kv-cache
python kv_cache.py
```

## Llama 3.1：从权重加载到文本生成

[llama31_inference.py](./10-llama-from-scratch/llama31_inference.py) 将完整推理链路集中在一个文件中，便于按顺序阅读：

```text
配置解析 → RoPE / GQA / RMSNorm / SwiGLU → Decoder Layer
        → Safetensors 分片权重加载 → Prefill → KV Cache 增量解码 → 采样输出
```

### 数值自检

使用 Tiny 配置，无需下载真实模型权重：

```bash
cd 10-llama-from-scratch
python llama31_inference.py --check
```

自检覆盖 RoPE、整模型 Logits、KV Cache 增量解码、Greedy 生成和采样复现，并在相应环节与 Hugging Face 实现进行对照。

### 真实模型推理

```bash
python llama31_inference.py \
  --prompt "用一句话解释 RoPE 是什么。" \
  --device cuda \
  --max-new-tokens 128 \
  --stats
```

默认从 ModelScope 获取 `LLM-Research/Meta-Llama-3.1-8B-Instruct`。也可以通过 `--model` 指定本地权重目录：

```bash
python llama31_inference.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --prompt "解释 KV Cache 的作用。" \
  --device cuda
```

真实 8B 模型对磁盘、内存或显存要求较高，请根据硬件条件选择运行方式；模型下载与使用还需遵守对应模型许可。

## 实现验证

仓库当前以**脚本内置自检和参考实现对照**为主，不同模块采用适合其内容的验证方式：

- **数值对齐**：Attention、SwiGLU、Llama 3.1 与 Hugging Face 参考实现对照；AdamW、学习率调度器和交叉熵与 PyTorch 实现对照。
- **一致性断言**：BPE 编解码与保存 / 加载往返检查；KV Cache 增量解码与全量重算结果检查。
- **形状与梯度检查**：ViT 输出形状、自定义 ReLU 梯度、RMSNorm 解析梯度等。
- **可视化实验**：RoPE 频率缩放、PreNorm / PostNorm、激活函数、学习率曲线和交叉熵相似度等。

部分参考实现对照依赖特定版本的 Transformers 接口。如遇接口差异，请先检查本地依赖版本，再定位对应模块的对齐代码。

## 配套讲解视频

理论篇侧重公式、直觉和推导，代码篇侧重实现细节与调试过程；尚未发布的模块不在表格中放置空链接。

| 模块 | 理论讲解 | 代码讲解 |
|:---|:---|:---|
| Backpropagation | 筹备中 | 筹备中 |
| RoPE | [旋转位置编码：理论篇](https://www.xiaohongshu.com/explore/6ad1ba3a00000000150052f8?xsec_token=ABQLRD8o_3HJsGCDwgRpJa1bf-p-8FKj3DAtyvIdiEeIw=&xsec_source=pc_user) | [旋转位置编码：代码篇](https://www.xiaohongshu.com/explore/6ad31eb3000000001a0211d2?xsec_token=ABKTNiA1JoS16skb7UebVOv_Rkrq_uTDf72VRh1ho_Ej4=&xsec_source=pc_user) |
| Self-Attention | [自注意力机制：理论篇](https://www.xiaohongshu.com/explore/6ad70acf000000002f03237d?xsec_token=ABj7eZjkQn7TJR4pfqLdYHEmtDECd8NPpmPz8YVDwI5cs=&xsec_source=pc_user) | [自注意力机制：代码篇](https://www.xiaohongshu.com/explore/6ad89540000000002c03b9e5?xsec_token=ABLnasY4rP1yhaEgpoGOe3pPNWXa2oSV_5fbys4lX9pGo=&xsec_source=pc_user) |
| RMSNorm | [均方根层归一化：理论篇](https://www.xiaohongshu.com/explore/6ae176b400000000340245ec?xsec_token=ABEUFAX6mC_j6-TeXgzNjC4bMFgzzrJJIDSGeHB6JS3mo=&xsec_source=pc_user) | [均方根层归一化：代码篇](https://www.xiaohongshu.com/explore/6af3d658000000002b010a94?xsec_token=ABls6L9vohmIqvtO3HjjVjAVn0065pAzCq-sFRUrZ95G0=&xsec_source=pc_user) |
| ReLU | [ReLU 激活函数：理论篇](https://www.xiaohongshu.com/explore/6ae5cd66000000001200b47b?xsec_token=ABZDUmuy1o3zs5tBuIW9VNi-2yYAlEoQTd2zymzTPHmpU=&xsec_source=pc_user) | 筹备中 |
| SwiGLU | [SwiGLU 激活函数：理论篇](https://www.xiaohongshu.com/explore/6ae730070000000006021a0b?xsec_token=YBHEE_PYJMEN8q-eGa1fBAl3RL1S1MyRByzo4coXDd14w=&xsec_source=pc_creatormng) | 筹备中 |
| AdamW | [AdamW 优化器：理论篇](https://www.xiaohongshu.com/explore/6ae81e580000000019002e2c?xsec_token=YBodqmf6y_frpJyAv4SP0qiiCCW5_hd38xbPaCI5fA4Qs=&xsec_source=pc_creatormng) | 筹备中 |

## 后续规划

项目会持续补充不同领域的 AI 基础手写算法。以下方向尚未在当前仓库中完整实现，欢迎通过 Issue 一起讨论优先级：

- **计算机视觉**：CNN、ResNet、目标检测、图像分割与视觉预训练。
- **多模态**：CLIP、跨模态注意力、图文检索、视觉语言模型与多模态对齐。
- **图像生成**：Autoencoder、VAE、GAN、U-Net 与图像生成质量评估。
- **Diffusion**：前向加噪与反向去噪、DDPM、DDIM、Classifier-Free Guidance、Latent Diffusion 和 Stable Diffusion 最小复现。
- **训练与对齐**：分布式训练、混合精度、梯度累积、Checkpoint、RLHF、DPO 与奖励模型。
- **推理与算子**：连续批处理、PagedAttention、量化、CUDA / Triton 算子。
- **工程质量**：统一依赖管理、自动化测试与持续集成。

## 参与贡献

欢迎提交 Issue 或 Pull Request。尤其欢迎以下类型的贡献：

- 修复公式、代码、注释或视频中的错误；
- 补充边界条件、数值测试和最小复现示例；
- 在保持可读性的前提下改进实现；
- 完善中文笔记、图示与模块讲解。

提交代码时，建议同时说明运行环境、验证方式和预期结果，避免引入仅能运行但难以阅读的黑盒封装。

## License

本项目基于 [MIT License](./LICENSE) 开源。第三方模型、权重和依赖库遵循其各自的许可协议。

## 致谢

感谢 [PyTorch](https://pytorch.org/)、[Hugging Face Transformers](https://github.com/huggingface/transformers) 与 [ModelScope](https://github.com/modelscope/modelscope) 等开源项目提供的参考实现和工具支持。

如果这个项目帮助你把某个抽象概念真正写成了代码，欢迎 Star、交流或提出改进建议。
