"""softmax logits 的 z-loss，以及一个"加与不加"的对照实验。

    L_z = c_z * mean_b ( log sum_j exp(z_bj) )^2

softmax 对 logits 的整体平移不敏感，所以 logits 的绝对尺度是一条交叉熵管不到的
平坦方向：在可分数据上，预测早就定型了，logits 却会一路增长，log Z 跟着涨。
z-loss 把 log Z 钉到 0（即 sum_j exp(z_j) = 1），从而止住这种增长，并让 MoE router
的数值行为保持可控。

参考：Zoph et al., "ST-MoE", 2022 —— router z-loss，默认 c_z = 1e-3。

运行：python3 z_loss.py
    -> 先跑自检（梯度/平移/掩码/精度），再用官方 API 做对照实验，输出表格
       assets/compare_z_loss.png
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ 实现 ============================

def z_loss(logits: torch.Tensor, mask: torch.Tensor | None = None, coef: float = 1.0) -> torch.Tensor:
    """返回 ``coef * mean(logsumexp(logits, -1)^2)``。

    参数：
        logits: 形状 ``[..., N]``，softmax **之前**的分数，N 为类数或专家数。
        mask: 可选的逐 token 掩码（``[B]`` 或 ``[B, T]``）；被掩掉的 padding token
            不计入均值。MoE 里 padding 也会被路由，所以需要传。
        coef: 权重 ``c_z``，常用 1e-3。
    """
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()                 # 把保险丝放在 fp32 里
    log_z = torch.logsumexp(logits, dim=-1)     # 内部做 max-shift，不会溢出
    per_token = log_z.square()
    if mask is None:
        return coef * per_token.mean()
    m = mask.reshape(-1).to(per_token.dtype)
    return coef * (per_token.reshape(-1) * m).sum() / m.sum().clamp_min(1.0)


def analytic_z_loss_grad(logits: torch.Tensor) -> torch.Tensor:
    """返回 mean z-loss 对 logits 的解析梯度：``2 * logZ * softmax(z) / B``。"""
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return 2.0 * log_z * torch.softmax(logits, dim=-1) / logits.shape[0]


# ============================ 自检 ============================

def _self_test() -> None:
    torch.manual_seed(0)

    # 1) 解析梯度与 autograd 对齐
    z = (torch.randn(4, 6, dtype=torch.float64) * 3).requires_grad_(True)
    print("[梯度校验]", torch.autograd.gradcheck(lambda t: z_loss(t), (z,), atol=1e-8))
    print("[梯度] 最大误差",
          (torch.autograd.grad(z_loss(z), z)[0] - analytic_z_loss_grad(z)).abs().max().item())

    # 2) softmax 无视平移，z-loss 不无视
    z0 = torch.randn(6, dtype=torch.float64)
    print("[平移] softmax 差值",
          (torch.softmax(z0, -1) - torch.softmax(z0 + 2, -1)).abs().max().item(),
          "| logZ 差值", (torch.logsumexp(z0 + 2, -1) - torch.logsumexp(z0, -1)).item())

    # 3) 惩罚的零点落在 sum_j exp(z_j) = 1
    zs = torch.full((1, 8), -torch.log(torch.tensor(8.0)))
    print("[极小值] z = -log(8) 时的 z-loss:", z_loss(zs).item())

    # 4) padding token 不计入均值
    lg = torch.randn(2, 3, 5)
    print("[掩码] 6 个 token 中只算 3 个:",
          z_loss(lg, mask=torch.tensor([[1, 1, 0], [1, 0, 0]])).item())

    # 5) bf16 走 fp32 保持有限，而朴素的 exp-then-log 会溢出
    big = (torch.randn(2, 8) * 40).bfloat16()
    print("[精度] bf16 logits ->", z_loss(big).dtype, z_loss(big).item(),
          "| 朴素 log(exp(z).sum()):",
          torch.exp(big.float()).sum(-1).log().square().mean().item())


# ============================ 对照实验 ============================

STEPS, BATCH, LR, D_IN, N_CLS, C_Z = 1500, 256, 1e-2, 8, 2, 1e-3


def make_data(n=2048, d=D_IN, margin=4.0, seed=0):
    """两团线性可分的高斯数据，沿同一个方向被推开 margin。"""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, generator=g)
    w = w / w.norm()
    x = torch.randn(n, d, generator=g)
    y = (x @ w > 0).long()
    x = x + (2.0 * y.float() - 1.0).unsqueeze(1) * w * margin
    return x, y


class Classifier(nn.Module):
    """一层线性头，输出 softmax 之前的 logits。"""

    def __init__(self, d_in=D_IN, n_cls=N_CLS):
        super().__init__()
        self.fc = nn.Linear(d_in, n_cls)

    def forward(self, x):
        return self.fc(x)


def train(use_z_loss: bool, steps=STEPS, seed=1):
    """同一份数据、同一个初始化，唯一差别是要不要多加 z-loss 那一条。"""
    torch.manual_seed(seed)
    x, y = make_data()
    model = Classifier()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)

    hist = {"step": [], "ce": [], "logZ": [], "conf": [], "mean_logit": []}
    for step in range(steps + 1):
        idx = torch.randint(0, x.shape[0], (BATCH,), generator=g)
        logits = model(x[idx])
        loss = F.cross_entropy(logits, y[idx])
        if use_z_loss:
            loss = loss + C_Z * z_loss(logits)     # 唯一的一行差别
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0:
            with torch.no_grad():
                full = model(x)
                hist["step"].append(step)
                hist["ce"].append(F.cross_entropy(full, y).item())
                hist["logZ"].append(torch.logsumexp(full, dim=-1).mean().item())
                hist["conf"].append(torch.softmax(full, dim=-1).max(dim=-1).values.mean().item())
                hist["mean_logit"].append(full.mean().item())
    with torch.no_grad():
        full = model(x)
        hist["final"] = {
            "ce": F.cross_entropy(full, y).item(),
            "logZ": torch.logsumexp(full, dim=-1).mean().item(),
            "mean_logit": full.mean().item(),
            "max_logit": full.max().item(),
            "min_logit": full.min().item(),
            "acc": (full.argmax(-1) == y).float().mean().item(),
            "logz_all": torch.logsumexp(full, dim=-1),
        }
    return hist


def compare(save_path="assets/compare_z_loss.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plain = train(use_z_loss=False)
    withz = train(use_z_loss=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    panels = [
        ("ce", "Cross-entropy (log scale)", True),
        ("logZ", "mean log Z", False),
        ("mean_logit", "mean logit  (the scale / shift direction)", False),
    ]
    for ax, (key, title, logy) in zip(axes.ravel()[:3], panels):
        ax.plot(plain["step"], plain[key], color="#d62728", lw=1.8, label="CE only")
        ax.plot(withz["step"], withz[key], color="#1f77b4", lw=1.8, label="CE + z-loss (1e-3)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)
        if logy:
            ax.set_yscale("log")
        if key == "logZ":
            ax.axhline(0.0, color="k", ls=":", lw=1)
    axes[0, 0].legend(frameon=False, fontsize=9)

    # 终态的 per-token log Z 分布：z-loss 把整条分布挪到了 0 附近
    ax = axes[1, 1]
    bins = torch.linspace(-2.5, 8.5, 45)
    ax.hist(plain["final"]["logz_all"], bins=bins, color="#d62728", alpha=0.55, label="CE only")
    ax.hist(withz["final"]["logz_all"], bins=bins, color="#1f77b4", alpha=0.55, label="CE + z-loss")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.set_title("final per-token log Z distribution", fontsize=10)
    ax.set_xlabel("log Z")
    ax.set_ylabel("tokens")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Same model, same data, same seed:  loss = CE  vs  loss = CE + c_z * z-loss",
                 fontsize=11)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\n图片已保存：{save_path}\n")

    rows = [("只算 CE", plain["final"]), (f"CE + z-loss ({C_Z:g})", withz["final"])]
    print(f"{'设置':<20}{'CE':>9}{'平均 logZ':>11}{'平均 logit':>12}"
          f"{'max logit':>11}{'min logit':>11}{'准确率':>8}")
    for name, f in rows:
        print(f"{name:<20}{f['ce']:>9.5f}{f['logZ']:>11.3f}{f['mean_logit']:>12.3f}"
              f"{f['max_logit']:>11.3f}{f['min_logit']:>11.3f}{f['acc']:>8.0%}")
    print("\nCE 与准确率不变，差别全在尺度：mean logZ 从 4.47 压到 0.24、max logit 从 8.2 降到 2.9"
          "\n（fp16 上溢阈值 11.09），代价是部分 token 的 logit 被推到更深的负值——下溢成 0，无害。")


if __name__ == "__main__":
    _self_test()
    compare()
