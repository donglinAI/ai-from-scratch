# -*- coding: utf-8 -*-
"""最终代码本地验证：四种变体与 nn.CrossEntropyLoss 官方逐项对齐"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyCustom(nn.Module):
    """自定义交叉熵：支持 weight / ignore_index / label_smoothing，与 nn.CrossEntropyLoss 对齐

    weight:           类别权重 [C]，给不同类别增加权重，放大少数类样本的损失贡献，让模型更关注少数类
    ignore_index:     忽略指定标签位置（如 padding / -100），该样本不参与 loss
    label_smoothing:  软标签平滑 ε；官方实现等价于真实类概率 1-ε+ε/C、其余类 ε/C
    """

    def __init__(self, weight=None, ignore_index=-100, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # logits: [B, C] raw scores, no softmax; targets: [B] class indices
        log_q = F.log_softmax(logits, dim=-1)
        mask = targets != self.ignore_index
        log_q, targets = log_q[mask], targets[mask]
        C = log_q.shape[-1]

        # 目标分布：one-hot 硬标签 或 平滑软标签
        if self.label_smoothing > 0.0:
            soft = torch.full_like(log_q, self.label_smoothing / C)
            soft[torch.arange(log_q.shape[0]), targets] = 1 - self.label_smoothing + self.label_smoothing / C
        else:
            soft = F.one_hot(targets, C).to(log_q.dtype)

        # 逐样本损失：-Σ_c p_c·log q_c；weight 按类别加权
        loss = -(soft * log_q).sum(-1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
            return loss.sum() / self.weight[targets].sum()  # 官方 mean：按权重和归一化
        return loss.mean()  # 在非忽略样本上平均


class FocalLoss(nn.Module):
    """Focal Loss：γ 压低易样本权重，α 缓解类别不均衡，支持 ignore_index"""

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma            # 调制因子，越大对高置信易样本压制越强
        self.alpha = alpha            # 类别权重 α_t，缓解类别不均衡
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        mask = targets != self.ignore_index
        log_q, targets = log_q[mask], targets[mask]

        logpt = log_q[torch.arange(log_q.shape[0]), targets]
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt  # (1-p_t)^γ 压低易样本
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 4, 5
    logits = torch.randn(B, C)
    targets = torch.tensor([0, 1, -100, 3])  # 位置 2 被 ignore_index 忽略
    w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # 基准 / weight / ignore_index / label_smoothing：与官方逐项对照
    print(f"CE           pytorch={nn.CrossEntropyLoss()(logits, targets):.6f} custom={CrossEntropyCustom()(logits, targets):.6f}")
    print(f"CE+weight    pytorch={nn.CrossEntropyLoss(weight=w)(logits, targets):.6f} custom={CrossEntropyCustom(weight=w)(logits, targets):.6f}")
    print(f"CE+ignore    pytorch={nn.CrossEntropyLoss(ignore_index=-100)(logits, targets):.6f} custom={CrossEntropyCustom(ignore_index=-100)(logits, targets):.6f}")
    print(f"CE+smoothing pytorch={nn.CrossEntropyLoss(label_smoothing=0.1)(logits, targets):.6f} custom={CrossEntropyCustom(label_smoothing=0.1)(logits, targets):.6f}")
    print(f"FocalLoss(γ=2, α=w) custom={FocalLoss(gamma=2.0, alpha=w)(logits, targets):.6f}")
