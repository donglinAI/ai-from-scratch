from __future__ import annotations

import torch 
import torch.nn as nn

import matplotlib.pyplot as plt 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim)) # 只有缩放gamma，没有偏移beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim], 在最后一维归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / var.add(self.eps).sqrt()
        # affine仿射变换
        return x_norm * self.gamma + self.beta
    
class SimpleSublayer(nn.Module):
    """模拟 Attention+MLP, 简化为两层线性+Gelu, 无bias、零均值初始化"""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)
        self.act = nn.GELU()
        # 标准零均值初始化
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class PreNormBlock(nn.Module):
    # x_{l+1} = x_{l} + Sublayer(Norm(x_{l}))
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.sub = SimpleSublayer(dim)

    def forward(self, x):
        residual = x 
        delta = self.sub(self.norm(x))
        return residual + delta
    

if __name__ == "__main__":
    torch.manual_seed(0)

    # 自动选择设备: cuda优先，否则cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dim, n_layers, batch_size, seq_len, vocab = 128, 16, 8, 16, 1000

    # 初始 embedding: 零均值初始化
    x = torch.randn(batch_size, seq_len, dim, device=device)
    norm = RMSNorm(dim).to(device)
    y1 = norm(x)
    y2 = norm(x * 2)
    y3 = norm(x + 2)
    # 缩放不变性证明
    print(f"归一化结果: {y1[0, 0, :5]}\n"
          f"缩放2倍归一化结果: {y2[0, 0, :5]}\n"
          f"平移+2归一化结果: {y3[0, 0, :5]}\n")

    
    blocks = nn.Sequential(
        *[PreNormBlock(dim) for _ in range(n_layers)]
    ).to(device)

    # 训练
    head = nn.Linear(dim, vocab, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(list(blocks.parameters()) + list(head.parameters()), lr=1e-4)
    
    for step in range(2000):
        x = torch.randn(batch_size, seq_len, dim, device=device) + 1
        target = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        opt.zero_grad()
        h = blocks(x)
        logits = head(h)
        loss = criterion(logits.reshape(-1,vocab), target.reshape(-1))
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step}, loss {loss.item():.3f}, hidden mean {h.mean().item():.4f}")

    mean_records = []
    rms_records = []
    with torch.no_grad():
        for l, blk in enumerate(blocks):
            # 记录进入该 block 前的 hidden 统计
            mu = x.mean().item()
            rms = torch.sqrt(x.pow(2).mean()).item()
            mean_records.append(mu)
            rms_records.append(rms)

            x = blk(x)
            
    # 绘图并保存
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(range(n_layers), mean_records, label="hidden mean")
    ax1.axhline(y=0, c="r", ls="--")  # 零均值参考线
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Mean of hidden state")
    ax1.set_title("Pre-RMSNorm Residual Stream: Mean over layers")
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig("residual_mean_train_layer.png", dpi=300)
    plt.close()