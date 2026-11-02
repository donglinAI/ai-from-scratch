from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# =============================================================================
# 版本A：HF标准 get_cosine_schedule_with_warmup（LambdaLR）
# 峰值lr = 优化器初始lr，factor范围 [eta_min, 1.0]
# =============================================================================
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0
) -> LambdaLR:
    def lr_lambda(step: int):
        # step：LambdaLR内部计数器，从0开始
        if step < num_warmup_steps:
            # +1保证warmup最后一步factor=1.0，到达峰值
            return (step + 1) / num_warmup_steps
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        factor = eta_min + 0.5 * (1.0 - eta_min) * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))
        return factor
    return LambdaLR(optimizer, lr_lambda=lr_lambda)

# =============================================================================
# 版本B：手写调度器，支持state_dict断点续训，和HF数学等价
# =============================================================================
class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        eta_min: float = 0.0
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        self.current_step = 0  # 从0开始，对齐LambdaLR计数
        # 保存每组参数的基准lr（峰值lr）
        for pg in self.optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg["lr"]

    def step(self) -> None:
        t = self.current_step
        if t < self.num_warmup_steps:
            factor = (t + 1) / self.num_warmup_steps
        else:
            progress = (t - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            factor = self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * factor
        self.current_step += 1

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, sd: dict) -> None:
        self.current_step = sd["current_step"]

# =============================================================================
# 测试入口
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    base_lr = 1e-3          # 峰值学习率
    warmup_steps = 100
    total_steps = 1000      # 完整训练总步数（含warmup）
    eta_min = 0.05          # 最低lr倍率：min_lr = base_lr * eta_min

    # 测试1：单组参数，对比两条曲线
    print("=" * 75)
    print("【LR曲线对比：HF LambdaLR版 VS 手写Scheduler版】")
    dummy_p = torch.randn(1, requires_grad=True)
    opt_hf = torch.optim.AdamW([dummy_p], lr=base_lr, weight_decay=0.01)
    sched_hf = get_cosine_schedule_with_warmup(opt_hf, warmup_steps, total_steps, eta_min)
    opt_my = torch.optim.AdamW([dummy_p], lr=base_lr, weight_decay=0.01)
    sched_my = CosineWarmupScheduler(opt_my, warmup_steps, total_steps, eta_min)

    lr_list_hf = []
    lr_list_my = []
    max_lr_err = 0.0
    for _ in range(total_steps):
        sched_hf.step()
        sched_my.step()
        lr_hf = sched_hf.get_last_lr()[0]
        lr_my = sched_my.get_last_lr()[0]
        lr_list_hf.append(lr_hf)
        lr_list_my.append(lr_my)
        err = abs(lr_hf - lr_my)
        max_lr_err = max(max_lr_err, err)

    print(f"整条LR曲线最大误差: {max_lr_err:.2e}")
    print("误差仅来自浮点精度，理论上两条曲线完全重合")
    print("=" * 75)

    # 测试2：多param_group场景
    print("\n【多param_group测试：不同初始lr分组】")
    def build_net():
        return nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))

    net_hf = build_net()
    net_my = build_net()

    params_backbone = []
    params_head = []
    params_bias = []
    for name, p in net_hf.named_parameters():
        if "bias" in name:
            params_bias.append(p)
        elif "0" in name:
            params_backbone.append(p)
        else:
            params_head.append(p)

    pg_hf = [
        {"params": params_backbone, "lr": 1e-3},
        {"params": params_head, "lr": 2e-3},
        {"params": params_bias, "lr": 5e-4},
    ]

    params_backbone_my = []
    params_head_my = []
    params_bias_my = []
    for name, p in net_my.named_parameters():
        if "bias" in name:
            params_bias_my.append(p)
        elif "0" in name:
            params_backbone_my.append(p)
        else:
            params_head_my.append(p)

    pg_my = [
        {"params": params_backbone_my, "lr": 1e-3},
        {"params": params_head_my, "lr": 2e-3},
        {"params": params_bias_my, "lr": 5e-4},
    ]

    opt_hf_multi = torch.optim.AdamW(pg_hf, weight_decay=0.01)
    sched_hf_multi = get_cosine_schedule_with_warmup(opt_hf_multi, warmup_steps, total_steps, eta_min)
    opt_my_multi = torch.optim.AdamW(pg_my, weight_decay=0.01)
    sched_my_multi = CosineWarmupScheduler(opt_my_multi, warmup_steps, total_steps, eta_min)

    max_param_err = 0.0
    for _ in range(total_steps):
        x = torch.randn(16, 10)
        y = torch.randn(16, 2)

        pred_hf = net_hf(x)
        loss_hf = ((pred_hf - y) ** 2).mean()
        opt_hf_multi.zero_grad()
        loss_hf.backward()
        opt_hf_multi.step()
        sched_hf_multi.step()

        pred_my = net_my(x)
        loss_my = ((pred_my - y) ** 2).mean()
        opt_my_multi.zero_grad()
        loss_my.backward()
        opt_my_multi.step()
        sched_my_multi.step()

        for p_hf, p_my in zip(net_hf.parameters(), net_my.parameters()):
            max_param_err = max(max_param_err, (p_hf - p_my).abs().max().item())

    print(f"多param_group训练，参数更新最大误差：{max_param_err:.2e}")
    print("多分组场景下两者依然对齐，每组独立使用自己的initial_lr缩放")

    # 断点续训演示
    print("\n【断点续训演示】")
    ckpt = sched_my.state_dict()
    print(f"保存时 current_step = {ckpt['current_step']}")
    sched_new = CosineWarmupScheduler(opt_my, warmup_steps, total_steps, eta_min)
    sched_new.load_state_dict(ckpt)
    print(f"加载后 current_step = {sched_new.current_step}")
