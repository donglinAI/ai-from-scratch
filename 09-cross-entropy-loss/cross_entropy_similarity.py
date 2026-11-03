import torch
import matplotlib.pyplot as plt


def cross_entropy(p, q):
    # 交叉熵 H(p,q) = -Σ p·log q；约定 0·log0 = 0（跳过 p=0 的项）
    p = torch.tensor(p, dtype=torch.float)
    q = torch.tensor(q, dtype=torch.float)
    mask = p > 0
    return -(p[mask] * torch.log(q[mask])).sum().item()


# 场景一：q 接近 p → 交叉熵小，分布相似
p_close = [0.7, 0.2, 0.1]
q_close = [0.68, 0.22, 0.10]
ce_close = cross_entropy(p_close, q_close)

# 场景二：q 远离 p → 交叉熵大，分布不相似
p_far = [0.7, 0.2, 0.1]
q_far = [0.10, 0.30, 0.60]
ce_far = cross_entropy(p_far, q_far)

print(f"H(p,q) close = {ce_close:.4f}  far = {ce_far:.4f}")

x = torch.arange(3).numpy()
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
for ax, p, q, ce, title in [
    (axes[0], p_close, q_close, ce_close, f"close (similar): H(p,q)={ce_close:.3f}"),
    (axes[1], p_far, q_far, ce_far, f"far (dissimilar): H(p,q)={ce_far:.3f}"),
]:
    ax.bar(x - width / 2, p, width, label="true p", color="#2277dd")
    ax.bar(x + width / 2, q, width, label="pred q", color="#dd4422", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["class 0", "class 1", "class 2"])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cross_entropy_similarity.png")