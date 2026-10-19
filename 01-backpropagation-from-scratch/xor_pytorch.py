

"""
用 PyTorch 解决 XOR 问题
网络结构：输入层(2) -> 隐藏层(3个神经元, tanh) -> 输出层(1, sigmoid)
"""

import torch
from torch import nn 
from torch import optim

# 1. 创建训练数据
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

X = torch.tensor(x_data, dtype=torch.float32)
y = torch.tensor(y_data, dtype=torch.float32)

# 2. 构建模型：输入层 + 隐藏层（3个神经元） + 输出层（1个神经元）
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.Tanh(),
    nn.Linear(3, 1),
    nn.Sigmoid(),
)

criterion = nn.MSELoss() # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1) # 优化器

# 3. 模型训练
for epoch in range(2000):
    optimizer.zero_grad()       # 梯度清零
    pred = model(X)             # 前向传播
    loss = criterion(pred, y)   # 计算损失
    loss.backward()             # 反向传播（自动求导）
    optimizer.step()            # 参数更新

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss = {loss.item():.6f}")

# 4. 模型推理
def predict(x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().ravel()

from utils import xor_solution_vis
xor_solution_vis.plot_xor_solution(X,y,predict,save_path="xor_solution_pth.png")