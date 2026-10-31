import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 模型定义 --------------------------
class BadReluNet(nn.Module):
    """对照组：Xavier初始化 + 无BN + ReLU"""
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # Xavier初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
    def get_all_preact(self, x):
        preacts = []
        tmp = x
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                tmp = layer(tmp)
                preacts.append(tmp)
            elif isinstance(layer, nn.ReLU):
                tmp = layer(tmp)
        return preacts


class OptimizedReluNet(nn.Module):
    """优化组：He初始化 + BN + ReLU"""
    def __init__(self, in_dim=16, hidden_dim=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # He初始化，适配ReLU
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)
    def get_all_preact(self, x):
        preacts = []
        tmp = x
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                tmp = layer(tmp)
                preacts.append(tmp)
            elif isinstance(layer, nn.BatchNorm1d):
                tmp = layer(tmp)
            elif isinstance(layer, nn.ReLU):
                tmp = layer(tmp)
        return preacts


# -------------------------- 工具函数：统计死神经元 --------------------------
def compute_dead_neuron_stats(preact_list):
    """
    preact_list: 每一层Linear输出的预激活张量 [B, H]
    返回：总死亡神经元数，每层平均激活率
    """
    total_dead = 0
    layer_act_rates = []
    for z in preact_list:
        act_rate = (z > 0.0).float().mean(dim=0)
        dead_mask = act_rate < 1e-5
        total_dead += dead_mask.sum().item()
        layer_act_rates.append(act_rate.mean().item())
    return total_dead, np.mean(layer_act_rates)


# -------------------------- 训练配置 --------------------------
in_dim = 16
hidden_dim = 128
batch_size = 256
train_steps = 100
lr = 1e-2

net_bad = BadReluNet(in_dim, hidden_dim).to(device)
net_opt = OptimizedReluNet(in_dim, hidden_dim).to(device)

opt_bad = optim.AdamW(net_bad.parameters(), lr=lr)
opt_opt = optim.AdamW(net_opt.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

# 固定训练数据分布
def get_batch():
    x = torch.randn(batch_size, in_dim, device=device)
    y = torch.randint(0, 2, (batch_size,), device=device)
    return x, y

# -------------------------- 训练对比 --------------------------
print(f"{'step':<5} | {'bad_dead':<8} | {'bad_act_mean':<12} | {'opt_dead':<8} | {'opt_act_mean':<12} | {'loss_bad':<10} | {'loss_opt':<10}")
print("-" * 80)

for step in range(train_steps):
    x, y = get_batch()

    # 对照组
    opt_bad.zero_grad()
    logits_bad = net_bad(x)
    loss_bad = loss_fn(logits_bad, y)
    loss_bad.backward()
    opt_bad.step()
    with torch.no_grad():
        pre_bad = net_bad.get_all_preact(x)
        dead_bad, act_bad = compute_dead_neuron_stats(pre_bad)

    # 优化组
    opt_opt.zero_grad()
    logits_opt = net_opt(x)
    loss_opt = loss_fn(logits_opt, y)
    loss_opt.backward()
    opt_opt.step()
    with torch.no_grad():
        pre_opt = net_opt.get_all_preact(x)
        dead_opt, act_opt = compute_dead_neuron_stats(pre_opt)

    if step % 5 == 0:
        print(f"{step:<5} | {dead_bad:<8d} | {act_bad:<12.4f} | {dead_opt:<8d} | {act_opt:<12.4f} | {loss_bad:<10.4f} | {loss_opt:<10.4f}")

# 输出结果
# step  | bad_dead | bad_act_mean | opt_dead | opt_act_mean | loss_bad   | loss_opt  
# --------------------------------------------------------------------------------
# 0     | 2        | 0.4764       | 0        | 0.4715       | 0.7204     | 0.8336    
# 5     | 14       | 0.3829       | 0        | 0.4838       | 0.7109     | 0.8668    
# 10    | 34       | 0.3788       | 0        | 0.4784       | 0.6940     | 0.7518    
# 15    | 52       | 0.3648       | 0        | 0.4695       | 0.6904     | 0.7777    
# 20    | 70       | 0.3501       | 0        | 0.4802       | 0.6977     | 0.7099    
# 25    | 73       | 0.3268       | 0        | 0.4667       | 0.6955     | 0.7072    
# 30    | 76       | 0.3259       | 0        | 0.4643       | 0.6959     | 0.7349    
# 35    | 81       | 0.2741       | 0        | 0.4715       | 0.6943     | 0.7324    
# 40    | 86       | 0.2841       | 0        | 0.4729       | 0.6943     | 0.7331    
# 45    | 83       | 0.2885       | 0        | 0.4621       | 0.6934     | 0.6905    
# 50    | 84       | 0.2874       | 0        | 0.4631       | 0.6941     | 0.7037    
# 55    | 89       | 0.2931       | 0        | 0.4573       | 0.6923     | 0.7093    
# 60    | 87       | 0.2909       | 0        | 0.4538       | 0.6967     | 0.6921    
# 65    | 84       | 0.2893       | 0        | 0.4525       | 0.6921     | 0.7092    
# 70    | 85       | 0.2879       | 0        | 0.4519       | 0.6928     | 0.6978    
# 75    | 89       | 0.2534       | 0        | 0.4481       | 0.6949     | 0.7053    
# 80    | 91       | 0.2788       | 0        | 0.4562       | 0.6938     | 0.7051    
# 85    | 87       | 0.2811       | 0        | 0.4410       | 0.6931     | 0.7022    
# 90    | 89       | 0.2946       | 0        | 0.4466       | 0.6937     | 0.7006    
# 95    | 88       | 0.2975       | 0        | 0.4496       | 0.6910     | 0.7018  