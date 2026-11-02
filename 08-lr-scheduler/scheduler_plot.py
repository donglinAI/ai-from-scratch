from __future__ import annotations
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import matplotlib.pyplot as plt


# 1. 纯余弦退火
def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    eta_min: float = 0.0
) -> LambdaLR:
    def lr_lambda(step: int):
        progress = step / num_training_steps
        factor = eta_min + 0.5 * (1.0 - eta_min) * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))
        return factor
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# 2. 线性预热+余弦退火（HF标准）
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0
) -> LambdaLR:
    def lr_lambda(step: int):
        if step < num_warmup_steps:
            return (step + 1) / num_warmup_steps
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        factor = eta_min + 0.5 * (1.0 - eta_min) * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))
        return factor
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# 3. 纯线性衰减
def get_linear_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    eta_min: float = 0.0
) -> LambdaLR:
    def lr_lambda(step: int):
        progress = step / num_training_steps
        factor = 1.0 - progress * (1.0 - eta_min)
        return factor
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# 4. 纯阶梯衰减
def get_step_schedule(
    optimizer: torch.optim.Optimizer,
    step_size: int,
    gamma: float = 0.5
) -> LambdaLR:
    def lr_lambda(step: int):
        decay_times = step // step_size
        factor = gamma ** decay_times
        return factor
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


if __name__ == "__main__":
    torch.manual_seed(42)
    base_lr = 1e-3
    warmup_steps = 100
    total_steps = 1000
    eta_min = 0.05
    step_size = 200
    gamma = 0.5

    dummy_p = torch.randn(1, requires_grad=True)

    # 初始化4个调度器
    opt_cos = torch.optim.AdamW([dummy_p], lr=base_lr)
    sched_cos = get_cosine_schedule(opt_cos, total_steps, eta_min)

    opt_cos_warm = torch.optim.AdamW([dummy_p], lr=base_lr)
    sched_cos_warm = get_cosine_schedule_with_warmup(opt_cos_warm, warmup_steps, total_steps, eta_min)

    opt_linear = torch.optim.AdamW([dummy_p], lr=base_lr)
    sched_linear = get_linear_schedule(opt_linear, total_steps, eta_min)

    opt_step = torch.optim.AdamW([dummy_p], lr=base_lr)
    sched_step = get_step_schedule(opt_step, step_size, gamma)

    lr_cos = []
    lr_cos_warm = []
    lr_linear = []
    lr_step = []

    for _ in range(total_steps):
        sched_cos.step()
        sched_cos_warm.step()
        sched_linear.step()
        sched_step.step()

        lr_cos.append(sched_cos.get_last_lr()[0])
        lr_cos_warm.append(sched_cos_warm.get_last_lr()[0])
        lr_linear.append(sched_linear.get_last_lr()[0])
        lr_step.append(sched_step.get_last_lr()[0])

    # 2行2列子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    axes[0].plot(lr_cos, lw=2)
    axes[0].set_title("cos")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("lr")
    axes[0].grid(alpha=0.25)

    axes[1].plot(lr_cos_warm, lw=2, color="#d62728")
    axes[1].axvline(x=warmup_steps, color="k", ls=":", alpha=0.6)
    axes[1].set_title("cos + warmup")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("lr")
    axes[1].grid(alpha=0.25)

    axes[2].plot(lr_linear, lw=2, color="#2ca02c")
    axes[2].set_title("linear")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("lr")
    axes[2].grid(alpha=0.25)

    axes[3].plot(lr_step, lw=2, color="#9467bd")
    axes[3].set_title("step")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("lr")
    axes[3].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("lr_subplot_compare.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ 子图对比图已保存：lr_subplot_compare.png")
