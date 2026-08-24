# ai-from-scratch

> 从零手动实现深度学习的一切：从反向传播到 Transformer，从图像分类到大模型 Agent。

## 关于这个项目

这是一个**以"从零手动实现"为核心的深度学习学习仓库**。

我相信，真正掌握一个算法的方式不是调包，而是亲手把它写出来。这个仓库记录了我从最基础的数学原理出发，一步步手动实现经典模型、大模型训练与 Agent 系统的全过程。每一个实现都遵循三个原则：

- **最小可运行**：去掉工程冗余，保留算法核心，几十行代码看清本质
- **注释详尽**：关键步骤逐行解释，方便对照论文理解推导过程
- **可复现**：提供数据下载、训练脚本和预期结果，拉下来就能跑

如果你也在学习深度学习，希望这个仓库能成为你的参考——不是因为它写得最好，而是因为它写得最"笨"、最透明。

## 覆盖范围

| 模块 | 内容 |
|---|---|
| **基础原理** | 反向传播、梯度下降、优化器（SGD/Adam）、归一化、激活函数 |
| **核心组件** | 全连接层、卷积层、循环层、注意力机制、Transformer Block |
| **经典模型** | LeNet、ResNet、LSTM、Transformer、GPT |
| **计算机视觉** | 图像分类、目标检测（YOLO/Faster R-CNN）、图像分割 |
| **多模态** | CLIP、图文对齐 |
| **大语言模型** | 预训练、SFT 微调、RLHF/DPO、推理加速 |
| **Agent** | Tool Use、ReAct 规划、多 Agent 协作 |
| **系统与工程** | 分布式训练、数据并行/张量并行/流水线并行、混合精度、KV Cache |

## 仓库结构

\`\`\`
ai-from-scratch/
├── README.md
├── 00-fundamentals/              # 数学与基础原理
├── 01-building-blocks/            # 神经网络核心组件
├── 02-classic-models/             # 经典模型从零实现
├── 03-computer-vision/            # 计算机视觉任务
├── 04-multimodal/                 # 多模态
├── 05-large-language-models/      # 大语言模型
├── 06-agent/                      # Agent 系统
├── 07-systems/                    # 系统与分布式
└── assets/                        # 图片 / 示意图
\`\`\`

## 学习路径

建议按以下顺序学习，由浅入深：

1. **\`00-fundamentals\`** → 理解梯度从哪来，优化器为什么有效
2. **\`01-building-blocks\`** → 搭积木，每一层的前向和反向都手写
3. **\`02-classic-models\`** → 用积木搭房子，理解经典架构设计
4. **\`03-computer-vision\` / \`04-multimodal\`** → 垂直领域应用
5. **\`05-large-language-models\`** → 预训练到对齐的完整流程
6. **\`06-agent\`** → 让模型从"说话"到"做事"
7. **\`07-systems\`** → 规模化训练的工程挑战

## 环境配置

\`\`\`bash
conda create -n ai-from-scratch python=3.10
conda activate ai-from-scratch
pip install torch numpy matplotlib tqdm
\`\`\`

## 进度追踪

- [ ] 00-fundamentals
- [ ] 01-building-blocks
- [ ] 02-classic-models
- [ ] 03-computer-vision
- [ ] 04-multimodal
- [ ] 05-large-language-models
- [ ] 06-agent
- [ ] 07-systems

## 致谢

感谢所有开源社区和论文作者。本仓库仅用于学习目的，所有实现参考对应论文与开源项目，具体引用见各子目录。

## License

[MIT](https://opensource.org/licenses/MIT)