from __future__ import annotations
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs("./images", exist_ok=True)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
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
        mu = x.mean(dim=-1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta

class SimpleSublayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.w2 = nn.Linear(4 * dim, dim, bias=False)
        self.act = nn.SiLU()
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class PreNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sub = SimpleSublayer(dim)
    def forward(self, x):
        return x + self.sub(self.norm(x))

class PostNormBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sub = SimpleSublayer(dim)
    def forward(self, x):
        return self.norm(x + self.sub(x))

class PostNormLNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.sub = SimpleSublayer(dim)
    def forward(self, x):
        return self.norm(x + self.sub(x))

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dim, n_layers, batch_size, seq_len, vocab = 128, 16, 8, 16, 1000

    # 缩放不变性
    rms_norm = RMSNorm(dim=dim).to(device=device)
    x = torch.randn(batch_size, seq_len, dim, device=device)
    rms_result1 = rms_norm(x)
    rms_result2 = rms_norm(x * 2)
    rms_result3 = rms_norm(x + 2)
    print(f"\n基准RMS归一化结果: {rms_result1[0, 0, :5].detach().cpu().numpy()}"
          f"\n缩放2倍归一化结果: {rms_result2[0, 0, :5].detach().cpu().numpy()}"
          f"\n平移2的归一化结果: {rms_result3[0, 0, :5].detach().cpu().numpy()}")

    # 3组模型
    pre_blocks = nn.Sequential(*[PreNormBlock(dim) for _ in range(n_layers)]).to(device)
    post_blocks = nn.Sequential(*[PostNormBlock(dim) for _ in range(n_layers)]).to(device)
    postln_blocks = nn.Sequential(*[PostNormLNBlock(dim) for _ in range(n_layers)]).to(device)
    head_pre = nn.Linear(dim, vocab, bias=False).to(device)
    head_post = nn.Linear(dim, vocab, bias=False).to(device)
    head_postln = nn.Linear(dim, vocab, bias=False).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        list(pre_blocks.parameters()) + list(head_pre.parameters())
        + list(post_blocks.parameters()) + list(head_post.parameters())
        + list(postln_blocks.parameters()) + list(head_postln.parameters()),
        lr=1e-4
    )

    pre_mean_rec = []
    post_mean_rec = []
    postln_mean_rec = []
    n_steps = 2000
    for step in range(n_steps):
        x = torch.randn(batch_size, seq_len, dim, device=device) + 1.0
        target = torch.randint(0, vocab, (batch_size, seq_len), device=device)
        opt.zero_grad()
        h_pre = pre_blocks(x)
        logits_pre = head_pre(h_pre)
        loss_pre = criterion(logits_pre.reshape(-1, vocab), target.reshape(-1))
        h_post = post_blocks(x)
        logits_post = head_post(h_post)
        loss_post = criterion(logits_post.reshape(-1, vocab), target.reshape(-1))
        h_postln = postln_blocks(x)
        logits_postln = head_postln(h_postln)
        loss_postln = criterion(logits_postln.reshape(-1, vocab), target.reshape(-1))
        loss = loss_pre + loss_post + loss_postln
        loss.backward()
        opt.step()
        if step % 20 == 0:
            pre_mean_rec.append(h_pre.mean().item())
            post_mean_rec.append(h_post.mean().item())
            postln_mean_rec.append(h_postln.mean().item())
            print(f"step {step:4d} | pre_mean {h_pre.mean():.4f} | post_rms_mean {h_post.mean():.4f} | post_ln_mean {h_postln.mean():.4f}")

    xs = np.arange(len(pre_mean_rec)) * 20
    fig, ax = plt.subplots(figsize=(10,5.5))
    ax.plot(xs, pre_mean_rec, label="PreNorm + RMSNorm")
    ax.plot(xs, post_mean_rec, label="PostNorm + RMSNorm")
    ax.plot(xs, postln_mean_rec, label="PostNorm + LayerNorm", linestyle="-.")
    ax.set_xlabel("training step")
    ax.set_ylabel("final hidden mean")
    ax.set_title("Mean evolution: PreNorm vs PostNorm(RMS) vs PostNorm(LN)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # ========== 修改点：输出SVG，开启压缩，减小体积 ==========
    plt.savefig(
        "./images/compare_mean_three_setting.svg",
        format="svg",
        bbox_inches="tight",
        metadata={},
        # 关闭多余渲染元素，svg内部精简
    )
    plt.close()
    print("\n✅ SVG矢量图已保存: ./images/compare_mean_three_setting.svg")
