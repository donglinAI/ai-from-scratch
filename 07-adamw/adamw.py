from __future__ import annotations 

import torch 
import torch.nn as nn


class AdamW:

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01,
    ):
        self.params = list(params)
        self.lr = lr 
        self.beta1, self.beta2 = betas 
        self.eps = eps 
        self.wd = weight_decay 
        # 每个参数独立的动量/方差/步数
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = [0] * len(self.params)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue 
            g = p.grad 
            self.t[i] += 1
            t = self.t[i]

            # 1) 解耦的权重衰减: 直接缩放参数（不进 m/v，系数是 lr 不是 step_size）
            p.data.mul_(1 - self.lr * self.wd)

            # 2) 一阶矩 / 二阶矩 （指数滑动平均）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g 

            # 3) Adam 更新: step_size 合并了一阶矩 bias correction 
            step_size = self.lr / (1 - self.beta1 ** t)
            denom = (self.v[i] / (1 - self.beta2 ** t)).sqrt() + self.eps 
            p.data.addcdiv_(self.m[i], denom, value=-step_size)


if __name__ == "__main__":

    # --- 手写测试 --- 
    print("手算演示: 单参数 θ=1.0, 固定梯度 g=0.1, 3步")
    p = torch.tensor([1.0], requires_grad=True)
    opt = AdamW([p], lr=0.1, weight_decay=0.01)
    for step in range(10):
        p.grad = torch.tensor([0.1])
        before = p.item()
        opt.step()
        print(f"Step {step+1}: θ {before:.4f} -> {p.item():.4f} | "
              f"m={opt.m[0].item():.4f} v={opt.v[0].item():.4f} t={opt.t[0]}")

    # --- 和 torch.optim.AdamW 对比 --- 
    torch.manual_seed(0)
    # 模型
    def make_model():
        return nn.Sequential(
            nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 4)
        )

    torch.manual_seed(0)
    my_model = make_model()
    torch.manual_seed(0)
    ref_model = make_model()

    opt_my = AdamW(my_model.parameters(), lr=1e-2, weight_decay=0.01)
    opt_ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-2, weight_decay=0.01)

    xs = [torch.randn(5, 6) for _ in range(10)]
    ys = [torch.randn(5, 4) for _ in range(10)]
    max_err = 0.0
    for x, y in zip(xs, ys):
        l1 = ((my_model(x) - y) ** 2).mean()
        opt_my.zero_grad(); l1.backward(); opt_my.step()

        l2 = ((ref_model(x) - y) ** 2).mean()
        opt_ref.zero_grad(); l2.backward(); opt_ref.step()

        for a, b in zip(my_model.parameters(), ref_model.parameters()):
            max_err = max(max_err, (a - b).abs().max().item())

    print(f"手写 AdamW 和 torch.optim.AdamW 实现的误差结果: {max_err}")

    