import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 采样点，开启梯度用于自动求导
x = torch.linspace(-4, 4, 1000, requires_grad=True)

# 前向计算
relu_y = F.relu(x)
sigmoid_y = torch.sigmoid(x)
silu_y = F.silu(x)

# 求梯度
relu_y.sum().backward(retain_graph=True)
relu_grad = x.grad.clone()
x.grad.zero_()

sigmoid_y.sum().backward(retain_graph=True)
sigmoid_grad = x.grad.clone()
x.grad.zero_()

silu_y.sum().backward(retain_graph=True)
silu_grad = x.grad.clone()
x.grad.zero_()

# SiLU函数最小值点
x_min_silu = torch.tensor(-1.278465)
y_min_silu = F.silu(x_min_silu)

# 创建1行2张子图，左右布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
x_np = x.detach().numpy()

# ========== 左图：激活曲线 ==========
ax1.plot(x_np, relu_y.detach().numpy(), color="blue", label="ReLU", linewidth=2)
ax1.plot(x_np, sigmoid_y.detach().numpy(), color="orange", label="Sigmoid(GLU Gate)", linewidth=2)
ax1.plot(x_np, silu_y.detach().numpy(), color="green", label="SiLU(SwiGLU Gate)", linewidth=2)

# 标注SiLU最小值
ax1.scatter(x_min_silu.item(), y_min_silu.item(), c="red", s=30, zorder=5)
ax1.annotate(
    f"min:({x_min_silu:.3f}, {y_min_silu:.3f})",
    xy=(x_min_silu.item(), y_min_silu.item()),
    xytext=(-2.2, -0.22),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=8,
    color="red"
)
ax1.axhline(y=0, color="gray", linestyle="--", lw=0.8)
ax1.axvline(x=0, color="gray", linestyle="--", lw=0.8)
ax1.set_xlabel("x")
ax1.set_ylabel("activation value")
ax1.set_title("Activation Curve")
ax1.legend()
ax1.grid(alpha=0.3)

# ========== 右图：梯度曲线 ==========
ax2.plot(x_np, relu_grad.detach().numpy(), color="blue", label="ReLU grad", linewidth=2)
ax2.plot(x_np, sigmoid_grad.detach().numpy(), color="orange", label="Sigmoid grad", linewidth=2)
ax2.plot(x_np, silu_grad.detach().numpy(), color="green", label="SiLU grad", linewidth=2)

ax2.axhline(y=0, color="gray", linestyle="--", lw=0.8)
ax2.axvline(x=0, color="gray", linestyle="--", lw=0.8)
ax2.set_xlabel("x")
ax2.set_ylabel("gradient")
ax2.set_title("Gradient Curve")
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle("ReLU vs Sigmoid(GLU) vs SiLU(SwiGLU)", fontsize=12)
plt.tight_layout()
plt.savefig("relu_glu_swiglu_act_grad.png", bbox_inches="tight")
plt.show()
