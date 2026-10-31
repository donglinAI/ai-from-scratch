import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 生成输入
x = torch.linspace(-4, 4, 1000, requires_grad=True)
relu = nn.ReLU()
y = relu(x)

# 逐点求梯度，避免梯度累加
grads = []
for xi in x:
    xi = xi.clone().detach().requires_grad_(True)
    yi = relu(xi)
    yi.backward()
    grads.append(xi.grad.item())

# 左右两张子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

# ReLU 前向曲线，颜色 blue
ax1.plot(x.detach().numpy(), y.detach().numpy(), color="blue", linewidth=2)
ax1.axhline(0, color="black", linewidth=0.5)
ax1.axvline(0, color="black", linewidth=0.5)
ax1.set_title("ReLU")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_ylim(-0.5, 4.5)

# ReLU 梯度曲线，颜色 red
ax2.plot(x.detach().numpy(), grads, color="red", linewidth=2)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.axvline(0, color="black", linewidth=0.5)
ax2.set_title("ReLU Gradient")
ax2.set_xlabel("x")
ax2.set_ylabel("gradient")
ax2.set_ylim(-0.2, 1.2)

plt.tight_layout()
plt.savefig("relu_two_subplot.png", dpi=300, bbox_inches="tight")
print("图片已保存：relu.png")
plt.show()
