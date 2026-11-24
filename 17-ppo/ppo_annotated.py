# ======================================================================================
# PPO（Proximal Policy Optimization）逐行注释版
# 来源：https://github.com/donglinAI/ai-from-scratch/blob/main/17-ppo/ppo.py
#
# 阅读方式建议：本文件只加注释，不改任何一行可执行代码（缩进、变量名、运算顺序全部原样），
# 因此可以拿它和原文件直接 diff 对照；跑起来的行为与原版完全一致。
#
# 一句话把握 PPO：用一个"不能太旧"的策略采一批数据（on-policy），
#   再用这批数据多训几遍（epochs），但每步都限制"新策略/旧策略"的比值 r 别跑太远，
#   于是既能样本复用，又不会一步跨崩。
#
# 四个关键量先记住，后面注释里反复出现：
#   r_t   = π_new(a|s) / π_old(a|s)     —— 新旧策略在同一动作上的概率比
#   A_t   —— 优势：这个动作比"平均水平"好多少
#   δ_t   = r_t + γV(s_{t+1}) - V(s_t)  —— 单步 TD 误差
#   L     = -min( r·A , clip(r,1±ε)·A ) —— 截断替代目标（取 min 是"悲观下界"）
# ======================================================================================

"""
PPO (Proximal Policy Optimization) —— PyTorch 从零实现，单文件可运行。
结构：
    1. ActorCritic    : 共享主干 + 策略头 + 价值头
    2. RolloutBuffer  : 采样存储
    3. compute_gae    : 广义优势估计
    4. ppo_update     : 截断目标（clipped surrogate）+ 价值回归 + 熵正则
    5. train          : on-policy 训练循环（每轮采样 → 多 epoch 更新 → 丢弃数据）
运行：
    pip install torch gymnasium
    python ppo.py                       # 默认 CartPole-v1，150 轮更新
    python ppo.py --updates 300 --plot assets/learning_curve.png
"""
# ↑ 模块级文档字符串（docstring）：用三引号写的说明，不参与运行，但 python -c "import ppo; help(ppo)" 能读到

import argparse     # 标准库：解析命令行参数（--updates 300 这种）
import os           # 标准库：建目录、拼路径（画图时保存文件要用）
import random       # 标准库：Python 自带随机数，用于给 random 模块设种子（影响环境内部的随机性）
import time         # 标准库：计时，打印"训练了多久"
from dataclasses import dataclass   # 标准库：用注解快速定义一个"纯数据配置类"，省掉手写 __init__

import numpy as np  # 数值库：环境返回的观测是 np.ndarray，转 tensor 前要过一手
import torch        # 深度学习框架：张量 + 自动求导
import torch.nn as nn           # 神经网络层（Linear / Tanh / Sequential）和初始化工具
import torch.nn.functional as F # 函数式接口，这里只用 F.mse_loss 算价值损失

try:  # gymnasium 为推荐依赖，gym 作为旧版兼容
    # gymnasium 是新版（0.28+ 起 reset 返回 (obs, info)、step 返回 5 元组），优先用它
    import gymnasium as gym
except ImportError:  # pragma: no cover
    # 装不上 gymnasium 就退回老版 gym；pragma: no cover 是给覆盖率工具看的"这行不测"
    import gym  # type: ignore

# --------------------------------------------------------------------------------------
# 超参数
# --------------------------------------------------------------------------------------
@dataclass                          # 装饰器：自动补 __init__/__repr__，下面每个"字段: 类型 = 默认值"就变成一个可写参数
class Config:                       # 所有超参集中在这里，调实验只改这个类（或命令行覆盖）
    env_id: str = "CartPole-v1"     # 环境名：倒立摆。观测 4 维（车位置/速度、杆角度/角速度），动作 2 个（左/右推）
    seed: int = 0                   # 随机种子：固定后同一份代码每次跑出来的曲线基本可复现
    total_updates: int = 150        # 迭代轮数（每轮采一批数据，更新完即丢弃）
                                    #   注意 PPO 是 on-policy：同一批数据用 cfg.epochs 遍后就作废，下一轮必须重新采样
    rollout_steps: int = 2048       # 每轮采样步数：一次交互收集 2048 个 (s,a,r,s') 存进 buffer
                                    #   2 的幂方便后续按 minibatch_size 切块
    gamma: float = 0.99             # 折扣因子 γ：未来奖励每步打折，0.99^100≈0.37，约 100 步后基本衰减完
    lam: float = 0.95               # GAE 的 lambda λ：偏差-方差旋钮。λ→0 只看一步 TD（低方差高偏差），λ→1 接近蒙特卡洛（高方差低偏差）
    clip_eps: float = 0.2           # PPO 截断范围 [1-eps, 1+eps]：概率比 r 超出这个区间就不再给梯度，防止策略一步跳太远
    lr: float = 3e-4                # Adam 学习率，PPO 常用 3e-4（比监督学习小，因为目标在动）
    epochs: int = 10                # 同一批数据重复训练轮数：样本复用效率的核心来源（但每 epoch 都重新打乱切 minibatch）
    minibatch_size: int = 64        # 每个小批 64 条，2048/64=32 个 minibatch，一 epoch 32 次梯度更新
                                    #   小批越小 → 更新越频繁但噪声越大
    value_coef: float = 0.5         # 价值损失权重 c1：总损失里价值项占多大分量
    entropy_coef: float = 0.01      # 熵奖励权重 c2：鼓励策略保持随机性，防止过早塌缩到一个动作（探索 vs 利用）
    max_grad_norm: float = 0.5      # 梯度全局范数裁剪阈值：防止某次异常大梯度把网络带崩
    target_kl: float = 0.015        # 近似 KL 超过该值提前结束本批更新：比 clip 更细的一道"策略漂移"闸门
    hidden_dim: int = 64            # 隐层宽度，CartPole 这么小的任务 64 足够
    eval_interval: int = 10         # 每 10 轮更新做一次贪心评测
    eval_episodes: int = 5          # 评测跑 5 局取平均，降低评测本身的随机波动
    log_interval: int = 1           # 每轮都打印训练日志
    device: str = "cpu"             # 设备：CartPole 用 CPU 就够，GPU 反而因传输开销更慢
                                    #   要上 GPU 就在命令行传 --device cuda

# --------------------------------------------------------------------------------------
# 1. 网络：Actor-Critic
# --------------------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """共享特征主干，两个输出头：动作分布 + 状态价值。
    - 策略头输出 logits，经 Categorical 得到动作分布 π(a|s)
    - 价值头输出标量 V(s)，只有 1 维
    """
    # ↑ 一个网络两份输出：既当"演员"（决定做什么）又当"评论家"（评估状态好坏）
    #   共享主干省参数、也让两头的特征表示互相受益
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        # obs_dim：观测维度（CartPole=4）；act_dim：离散动作个数（CartPole=2）
        super().__init__()          # 必须先调父类构造：nn.Module 靠它初始化参数注册系统，否则 self.xxx = nn.Linear 会报错
        self.trunk = nn.Sequential( # 共享主干：两层 MLP，把原始观测映射成 64 维特征 h
            nn.Linear(obs_dim, hidden_dim),   # 4 → 64 线性变换（Wx+b）
            nn.Tanh(),                        # 激活：PPO 惯用 tanh（输出有界 [-1,1]，比 ReLU 对策略输出更平滑）
            nn.Linear(hidden_dim, hidden_dim),# 64 → 64，再来一层增加表达力
            nn.Tanh(),                        # 再过一次激活，输出 h 同时喂给两个头
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)  # 策略头：h → logits（每个动作一个"未归一化分数"）
        self.value_head = nn.Linear(hidden_dim, 1)         # 价值头：h → 1 个数，即 V(s) 的估计
        # 正交初始化：策略头小增益输出接近均匀分布，价值头增益 1.0
        # ↑ 正交初始化让不同神经元的权重方向尽量"正交"，信号在层间传播时不会被放大或衰减
        for m in self.trunk:                          # 遍历主干的每个子模块
            if isinstance(m, nn.Linear):              # 只处理线性层（Tanh 没有参数可初始化）
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))  # 权重正交化，增益 √2 ≈ 1.414
                #   tanh 的线性区导数≈1，乘 √2 让前向方差大致保持不变（等价于"适合 tanh 的 He 初始化"）
                nn.init.constant_(m.bias, 0.0)        # 偏置置零，保持初始化时无额外偏移
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)  # 策略头增益极小 → logits≈0
        #   logits 全 0 时 softmax 是均匀分布：训练起点"完全不知道该选什么"，探索性最强，也不会一开始就锁死在某个动作
        nn.init.constant_(self.policy_head.bias, 0.0)            # 偏置同样清零，保证起点严格均匀
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)    # 价值头增益 1.0：输出量级直接对标回报量级，不用刻意缩
        nn.init.constant_(self.value_head.bias, 0.0)             # 偏置 0：初始 V(s)≈0，前期优势 A≈回报，信号最强

    def forward(self, obs: torch.Tensor):
        """返回 (动作分布, 状态价值)。"""
        # obs 形状 [B, obs_dim]，B 为批大小（可以是 1 也可以是 64）
        h = self.trunk(obs)                          # 共享特征：[B, obs_dim] → [B, hidden_dim]
        logits = self.policy_head(h)                 # 策略 logits：[B, hidden_dim] → [B, act_dim]
        dist = torch.distributions.Categorical(logits=logits)  # 包成离散分布对象
        #   内部会做 log_softmax，所以直接传 logits 比先 softmax 再传 probs 数值更稳（避免 log(0)）
        #   有了 dist 就同时拿到 .sample()（采样）、.log_prob(a)（对数概率）、.entropy()（熵）、.probs（概率）
        value = self.value_head(h).squeeze(-1)       # 价值：[B, hidden_dim] → [B, 1] → squeeze → [B]
        #   squeeze(-1) 去掉最后一维：后续算 loss 时要和 [B] 形状的 returns 对齐，避免广播出 [B,B]
        return dist, value                           # 一次前向同时给出"怎么选"和"这状态值多少"

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """采样阶段：按概率分布采样动作（不是 argmax），保留探索性。"""
        # @torch.no_grad()：采样不需要梯度，关掉自动求导省显存、也避免采样图被误保留
        dist, value = self.forward(obs)              # 前向拿到分布和价值
        action = dist.sample()                       # 按 π(a|s) 真采样：这是 on-policy 的关键——动作必须来自当前策略本身
        #   如果这里用 argmax，行为策略就变成确定性的贪心，和要优化的随机策略不一致，"重要性采样比" r 就失真
        logp = dist.log_prob(action)                 # 记录采样时刻的 log π_old(a|s)，存在 buffer 里当"旧概率"基准
        return action, logp, value                   # 动作 / 旧对数概率 / 当前状态价值，三样都要存进 buffer

# --------------------------------------------------------------------------------------
# 2. 采样缓存
# --------------------------------------------------------------------------------------
class RolloutBuffer:
    # ↑ 只做一件事：把 2048 步的交互记录先存成 Python list，最后一次性转 tensor
    #   为什么不在线转：每步都建 tensor 开销大且碎；先攒 list 再批量 stack 更省
    def __init__(self):
        self.obs, self.actions, self.logprobs = [], [], []          # 状态 s / 动作 a / 旧策略对数概率 logπ_old(a|s)
        self.rewards, self.values, self.values_next, self.terminals = [], [], [], []
        #   rewards r / 当前状态价值 V(s) / 下一状态价值 V(s') / 是否真终止（1.0 或 0.0）
        #   后三样是 GAE 的必要输入：算 δ=r+γV(s')-V(s) 时缺一不可

    def add(self, obs, action, logp, reward, value, value_next, terminal):
        # 每与环境交互一步就调一次，把这一"过渡"(transition)的 7 个字段各追加进对应 list
        self.obs.append(obs)                 # 当前状态 s_t
        self.actions.append(action)          # 实际执行的动作 a_t（int，之后转 long 做索引）
        self.logprobs.append(logp)           # 采样时刻的 log π_old(a_t|s_t)，更新时用来算概率比
        self.rewards.append(reward)          # 这一步拿到的即时奖励 r_t
        self.values.append(value)            # V(s_t)：采样时前向得到的价值，作为 GAE 的"起点基线"
        self.values_next.append(value_next)  # V(s_{t+1})：用于自举；真终止时填 0
        self.terminals.append(terminal)      # 1.0=真终止（episode 结束），0.0=未终止
        #   注意：时间上限截断（truncated）不算真终止，见 rollout 里的注释

    def stack(self, device):
        # 采样结束调用一次：把 7 个 list 全转成同设备、同 dtype 的 tensor，供向量化计算
        def t(x):
            # 局部小函数：list → tensor 的统一转换管道
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
            #   np.asarray 先把"list of ndarray / list of float"统一成二维或一维 ndarray
            #   再 as_tensor 共享内存转 tensor（torch.as_tensor 对 ndarray 是零拷贝，比 torch.tensor 省一次复制）
            #   dtype 固定 float32：与网络参数一致，避免 float64 混算报错/降速
        obs = t(self.obs)                    # [N, obs_dim]，N=rollout_steps
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device)
        #   ↑ 动作必须 long(int64)：Categorical.log_prob 要用它做索引查表，float 会直接报错
        logprobs = t(self.logprobs)          # [N]  旧对数概率
        rewards = t(self.rewards)            # [N]  奖励
        values = t(self.values)              # [N]  V(s_t)
        values_next = t(self.values_next)    # [N]  V(s_{t+1})（终止处为 0）
        terminals = t(self.terminals)        # [N]  0/1 终止标记
        return obs, actions, logprobs, rewards, values, values_next, terminals
        # 按顺序返回，调用方用一行解包，字段顺序就是"契约"

# --------------------------------------------------------------------------------------
# 3. GAE：广义优势估计
# --------------------------------------------------------------------------------------
def compute_gae(rewards, values, values_next, terminals, gamma, lam):
    r"""逐时刻倒序递推：
    δ_t = r_t + γ V(s_{t+1})(1-done_t) - V(s_t)
    A_t = δ_t + γλ(1-done_t) A_{t+1}
    """
    # ↑ 函数名前的 r 不是"转义"，是告诉 Python 这是 raw string，里面的 \ 不当转义符（本函数其实没用反斜杠，习惯写法）
    #   优势 A(s,a)=Q(s,a)-V(s)：衡量"选这个动作比按策略平均水平好多少"
    #   直接估 Q 方差太大，GAE 用 δ（一步 TD 误差）的加权和来折中偏差与方差
    n = rewards.shape[0]                                  # 序列长度 N（这里是 2048）
    advantages = torch.zeros_like(rewards)                # 预分配同形状全 0 张量，逐位回填 A_t
    last_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    # ↑ 标量 0（形状 [] 的 0 维张量），代表 A_{t+1}；放在最后一步之后时它的值是 0（episode 之后没有未来）
    #   必须用带 dtype/device 的 torch.zeros 而非 Python 的 0.0，否则下一步与张量相乘会引入类型提升/设备不一致
    for t in reversed(range(n)):
        # 倒序循环：因为 A_t 依赖 A_{t+1}，只能从后往前算（正序算不出未来）
        mask = 1.0 - terminals[t]                         # 掩码：未终止=1，真终止=0 → 用来"一键掐断"跨 episode 的自举
        #   为什么要掐断：终止后的 V(s') 已无意义，若仍乘 γV(s')，等于把下一局的回报错误地算进本局
        delta = rewards[t] + gamma * values_next[t] * mask - values[t]
        #   δ_t = r_t + γ·V(s_{t+1})·mask - V(s_t)
        #   mask=0 时 δ_t 退化为 r_t - V(s_t)：终止处没有"未来"，只剩这一步的实际收益
        #   注意这里用的是采样时缓存的 V(s_{t+1})（旧价值），不是重新前向算的——保证 GAE 目标在整个 epoch 内固定不变
        last_adv = delta + gamma * lam * mask * last_adv
        #   A_t = δ_t + γλ·mask·A_{t+1}：把后面的 δ 按 (γλ)^k 权重累加回来
        #   λ 越大，越像"看到 episode 结束"的蒙特卡洛；λ 越小，越像"只看一步"的 TD
        advantages[t] = last_adv                          # 存进对应位置
    returns = advantages + values  # 价值网络回归目标
    # ↑ return 的两种等价写法：A_t + V(s_t) 或 δ 的 γ 折扣累加
    #   为什么用 A+V 而不是直接累加奖励：A 已经由 GAE 平滑过，A+V 天然更稳、方差更小
    #   价值头回归的就是这个 returns（用采样时的 V 参与构造，但作为"标签"不参与梯度）
    return advantages, returns                            # 优势用于策略损失，returns 用于价值损失

# --------------------------------------------------------------------------------------
# 4. PPO 更新
# --------------------------------------------------------------------------------------
def ppo_update(model, optimizer, buffer, cfg):
    obs, actions, old_logprobs, rewards, values, values_next, terminals = buffer.stack(cfg.device)
    # ↑ 把 2048 步采样记录整体转成 tensor，搬到目标设备
    advantages, returns = compute_gae(rewards, values, values_next, terminals, cfg.gamma, cfg.lam)
    # ↑ 先算好两个"固定标签"：优势 A（策略损失用）、回报 returns（价值损失用）
    #   关键：这两者在下面的 epochs 循环里始终不变 → 同一批数据可以反复训而不漂移
    # 优势标准化：与 GRPO 的组内归一化同理，都是把"信号"变成可比的分差
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # ↑ 减均值除标准差 → 让 A 大致服从 N(0,1)
    #   好处：不同任务/不同奖励尺度下，梯度量级稳定，学习率不用跟着奖励量级重调
    #   1e-8 防除零：全部优势恰好相同时 std=0，不加会出 inf/nan
    #   副作用：标准化后"平均优势为 0"，等于做了一次基线校正，进一步降方差
    batch_size = obs.shape[0]                              # 2048
    idx = torch.arange(batch_size, device=cfg.device)      # 生成 0..2047 的索引向量，下面打乱切块都用它
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "n": 0}
    # ↑ 累积统计量，用来算整个更新阶段的均值并打印/监控；n 记录做了多少次 minibatch 更新
    for _ in range(cfg.epochs):                            # 同一批数据重复 10 遍（样本复用）
        idx = idx[torch.randperm(batch_size, device=cfg.device)]
        # ↑ 每个 epoch 重新打乱一次：让 minibatch 的组成随机化，避免固定顺序带来的相关性偏差
        #   torch.randperm 生成随机排列，再拿它去索引 idx → 得到打乱后的下标序列
        for start in range(0, batch_size, cfg.minibatch_size):   # 步长 64：0,64,128,...，共 32 个 minibatch
            mb = idx[start:start + cfg.minibatch_size]           # 取出这一小批的样本下标（一个 1-D long 张量）
            dist, new_values = model(obs[mb])                    # 用"当前（正在变新的）策略"前向这一小批
            #   obs[mb] 是高级索引：按 mb 里的一串下标挑出对应行 → [64, obs_dim]
            new_logprobs = dist.log_prob(actions[mb])            # 新策略下这批动作的对数概率 logπ_new(a|s)
            entropy = dist.entropy().mean()                      # 熵：分布越接近均匀越大，是"探索度"的度量
            #   .mean() 对 64 个样本取平均，得到一个标量，作为正则项加进损失
            log_ratio = new_logprobs - old_logprobs[mb]          # log r = logπ_new - logπ_old（对数域相减比直接除更稳）
            ratio = log_ratio.exp()                              # r = π_new/π_old：>1 表示这个动作现在更可能被选
            adv = advantages[mb]                                 # 这批样本的优势（固定标签，无梯度）
            # 未裁剪目标 与 裁剪目标，取较小者 → 悲观下界
            surr1 = ratio * adv                                  # 原始重要性采样目标：r·A（r 可无界，一步就能把策略推到天边）
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            # ↑ 截断版：先把 r 夹到 [0.8, 1.2] 再乘 A。r 超出区间时梯度为 0，相当于"这一步不再推动"
            policy_loss = -torch.min(surr1, surr2).mean()
            # ↑ 取 min 是核心技巧，分两种情况理解（这是理解 PPO 的关键，务必对上 A 的符号）：
            #   A>0（好动作，希望提高其概率）：r 涨过 1+ε 时 min 选 surr2(常数) → 梯度 0，不再奖励继续涨
            #   A<0（坏动作，希望降低其概率）：r 掉到 1-ε 以下时 min 选 surr2(常数) → 梯度 0，不再奖励继续降
            #   于是"想推的方向"被夹住，"想缩的方向"永远不拦——这就是 clip 的精确语义：只限制朝有利方向过冲
            #   前面加负号：优化器做的是梯度下降，而我们想最大化目标 → 取负变成最小化
            value_loss = F.mse_loss(new_values, returns[mb])
            # ↑ 价值头就是回归任务：预测 V(s) 去拟合 returns 标签，均方误差。
            #   returns 是常数标签（.detach 效果天然具备，因为它是从 buffer 转来的、不带计算图），
            #   所以梯度只从 new_values 流回价值头，不会去动策略那一路
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            # ↑ 总损失 = 策略(-目标) + 0.5×价值MSE - 0.01×熵
            #   熵前面是负号：我们要"最大化熵"，而在最小化 loss，所以减
            #   三项的物理意义：把好动作的概率调高 + 把 V 估准（好当基线）+ 别让策略过早变确定
            optimizer.zero_grad(set_to_none=True)                # 清空上一步的梯度
            #   set_to_none=True 把 .grad 置为 None 而非填 0：省内存、也让 PyTorch 少做一次遍历写零
            loss.backward()                                      # 反向传播：链式求导算出每个参数的梯度
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            # ↑ 按"全局 L2 范数"裁剪梯度：把所有权重梯度拼成一个大向量，范数超过 0.5 就等比缩小
            #   作用：单个异常 minibatch（比如突然一局崩了）不会造成一次毁灭性更新；注意它改的是方向长度不改方向
            optimizer.step()                                     # Adam 用裁剪后的梯度更新参数
            # 近似 KL（无偏、恒正）：(r-1) - log r
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            # ↑ 这是 KL(π_old‖π_new) 的二阶泰勒展开估计（常称 k3 估计）
            #   展开验证：r=e^x，x=log r 很小时 e^x-1-x ≈ x²/2 ≥ 0，所以它恒非负
            #   为什么不直接算 true KL：要遍历全部动作求和，成本高；这个只用已采样的动作，无偏且省
            #   .item() 把 0 维张量转成 Python float，避免把标量留在计算图里
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
            # ↑ 被截断样本的占比：衡量"有多少步撞到了夹子"，健康训练里应保持在小比例
            #   如果一路接近 100%，说明学习率太大或 epochs 太多，策略每次都被顶在墙上
            stats["policy_loss"] += policy_loss.item()           # 累加各项标量，最后取均值
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n"] += 1                                      # 计数 +1，作为求均值的分母
        if stats["n"] and stats["kl"] / stats["n"] > cfg.target_kl:
            break  # 新旧策略偏离过大，提前停止本批更新
            # ↑ 每个 epoch 结束后检查"到目前为止的平均 KL"是否超阈值 0.015
            #   超了就跳出 epochs 循环，剩下的 epoch 不再训 —— 宁可少用数据也不让策略跑偏
            #   这是比 clip 更外层的保险：clip 管单步，target_kl 管整批累积漂移
    for k in ("policy_loss", "value_loss", "entropy", "kl", "clip_frac"):
        stats[k] = stats[k] / max(stats["n"], 1)                 # 逐项求均值；max(n,1) 防 n=0 时除零
    return stats, advantages                                     # 返回统计字典 + 标准化后的优势（可用于调试/画图）

# --------------------------------------------------------------------------------------
# 5. 环境交互（兼容 gymnasium / gym 两套 reset-step 接口）
# --------------------------------------------------------------------------------------
def env_reset(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()   # 有种子就带种子重置，保证初始状态可复现
    if isinstance(out, tuple):                                        # 新版 gymnasium 返回 (obs, info) 二元组
        return out[0], out[1]                                         # 拆开：只要 obs，info 也返回给调用方
    return out, {}                                                    # 老版 gym 只返回 obs → 补一个空 info，保持函数签名统一

def env_step(env, action):
    out = env.step(action)                                            # 推进一步
    if len(out) == 5:                                                 # 新版返回 5 元组（把 done 拆成了 terminated + truncated）
        obs, reward, terminated, truncated, info = out                # terminated=真结束（如杆倒了）；truncated=被时间上限截断（如 500 步）
        return obs, float(reward), bool(terminated), bool(truncated), info   # 统一转成 Python 原生类型，避免 numpy 标量混进 tensor 运算
    obs, reward, done, info = out                                     # 老版 4 元组：只有一个 done
    return obs, float(reward), bool(done), False, info                # 把 done 当 terminated，truncated 恒 False，接口对齐

def obs_to_tensor(obs, device):
    return torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
    # ↑ .unsqueeze(0)：单个观测形状 [4] → [1, 4]，补一个 batch 维
    #   因为网络 forward 默认输入是 [B, obs_dim]，batch=1 也不能少这一维，否则矩阵乘维数对不上

@torch.no_grad()                                                     # 评测只推理不训练，关掉梯度省算力
def evaluate(model, env, cfg):
    model.eval()                                                     # 切到 eval 模式（本网络没有 BN/Dropout，这步主要是规范写法，防以后加层出错）
    rewards = []                                                     # 存每局的累计回报
    for _ in range(cfg.eval_episodes):                               # 默认跑 5 局
        obs, _ = env_reset(env)                                      # 每局开始重置（评测不固定种子，看的是平均泛化表现）
        done, ep_ret = False, 0.0                                    # 局结束标记 + 本局累计回报
        while not done:                                              # 一直走到 episode 结束
            dist, _ = model(obs_to_tensor(obs, cfg.device))          # 前向拿动作分布（价值那头用不上，用 _ 丢掉）
            action = int(dist.probs.argmax().item())  # 评测用贪心   # 评测取概率最大的动作：排除采样噪声，看"策略本身多强"
            obs, reward, terminated, truncated, _ = env_step(env, action)
            done = terminated or truncated                           # 两种结束都算本局结束，否则会死循环
            ep_ret += reward                                         # 累加（应为未折扣的原始回报，便于和 reward_threshold 比较）
        rewards.append(ep_ret)                                       # 本局得分入列表
    model.train()                                                    # 评测完切回 train 模式，别忘了
    return float(np.mean(rewards))                                   # 5 局取均值，作为一个稳定的评测分数

@torch.no_grad()                                                     # 采样阶段也不需要梯度
def rollout(model, env, obs, cfg, ep_returns):
    """采样 rollout_steps 步；返回 buffer 与新起点 obs。
    ep_returns 是长度至少为 1 的列表：最后一个元素是"正在进行中"这一局
    的累计回报，每结束一局就 append 一个新的 0.0。
    """
    # ↑ 这个设计解决了"跨 update 的 episode"问题：update 边界不一定是 episode 边界，
    #   所以用 ep_returns 列表在跨轮次传递"未完结那一局的分数"，等它真结束再结算
    buf = RolloutBuffer()                                            # 新建空 buffer，本轮的 2048 步全存这里
    for _ in range(cfg.rollout_steps):                               # 固定采 2048 步（可能跨好几个 episode）
        o_t = obs_to_tensor(obs, cfg.device)                         # 当前观测 → [1, obs_dim]
        action, logp, value = model.act(o_t)                         # 按当前策略采样：动作 / 旧 log 概率 / V(s_t)
        a = int(action.item())                                       # 0 维张量 → Python int，环境 step 需要原生 int
        next_obs, reward, terminated, truncated, _ = env_step(env, a)  # 环境推进一步
        # 只有真终止才是"价值为 0"；truncated（时间上限）仍要自举
        terminal = float(terminated)                                 # 只把"真终止"记为 1；CartPole 撞到 500 步上限是 truncated，不算
        #   ↑ 这是 RL 里一个高频坑：写 float(terminated or truncated) 会把时间截断当死亡，
        #     导致 500 步时的 V(s') 被错误置零，价值估计系统性偏低（因为下一局其实还能继续产生回报）
        v_next = 0.0                                                 # 默认下一状态价值为 0（终止时用）
        if not terminated:                                           # 未终止（含 truncated）才需要真实算 V(s')
            _, v_next_t = model(obs_to_tensor(next_obs, cfg.device)) # 前向算下一状态价值（只用价值头，丢弃 dist）
            v_next = float(v_next_t.item())                          # 转成 Python float 存进 buffer
        buf.add(obs, a, float(logp.item()), reward, float(value.item()), v_next, terminal)
        # ↑ 存一条完整 transition。注意 logp/value 都 .item() 成 float 再存：
        #   buffer 里全是"纯数值"，更新阶段重新前向算新概率，两边不发生任何计算图牵连
        ep_returns[-1] += reward                                     # 把奖励累进"当前进行中这一局"的分数
        if terminated or truncated:                                  # 一局结束（无论真终止还是超时）
            ep_returns.append(0.0)                                   # 为下一局开一个 0 的槽位
            obs, _ = env_reset(env)                                  # 环境重置，回到初始状态
        else:
            obs = next_obs                                           # 没结束就顺着往下走
    return buf, obs                                                  # 返回采样结果 + 最新的观测（下一轮接着采，不从零开始）

# --------------------------------------------------------------------------------------
# 6. 训练主循环
# --------------------------------------------------------------------------------------
def train(cfg):
    random.seed(cfg.seed)                                            # 三个随机源全都设种子：Python 随机 / numpy 随机 / torch 随机
    np.random.seed(cfg.seed)                                         #   三者各自独立，少设一个就做不到完全复现
    torch.manual_seed(cfg.seed)                                      #   注意：这只管 CPU；GPU 还要 torch.cuda.manual_seed_all
    env = gym.make(cfg.env_id)                                       # 训练环境
    eval_env = gym.make(cfg.env_id)                                  # 单独建一个评测环境，避免评测干扰训练环境的内部状态
    obs_dim = env.observation_space.shape[0]                         # 观测维度 = 4（Box 空间的 shape）
    act_dim = env.action_space.n                                     # 动作个数 = 2（Discrete 空间的 n）
    model = ActorCritic(obs_dim, act_dim, cfg.hidden_dim).to(cfg.device)   # 建网络并搬到设备
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    # ↑ Adam 自适应学习率。eps=1e-5 比默认 1e-8 大得多是有意为之：
    #   PPO 早期梯度可能很小，1e-8 会让 1/sqrt(v) 爆炸，1e-5 起到"分母下限"的稳定作用（RL 社区的常规做法）
    obs, _ = env_reset(env, seed=cfg.seed)                           # 第一次重置，指定种子 → 起始状态确定
    best_eval = -float("inf")                                        # 记录历史最优评测分，用负无穷做初值，保证第一次一定刷新
    history = []                                                     # 每轮记录 (update, 平均回报, 统计字典)，最后拿去画曲线
    ep_returns = [0.0]           # 当前 episode 累计回报（末尾为进行中的 episode）
    finished_returns = []                                            # 已结束 episode 的分数总账
    start = time.time()                                              # 训练开始计时

    for update in range(1, cfg.total_updates + 1):                   # 主循环：150 轮
        n_before = len(finished_returns)                             # 记下本轮开始前的已完成局数，用来切出"本轮新完结的局"
        buf, obs = rollout(model, env, obs, cfg, ep_returns)         # 采 2048 步；obs 被更新，下一轮接着采
        # 结算本轮完成的 episode
        finished_returns.extend(ep_returns[:-1])                     # 除最后那个"进行中"的槽位，其余都已有最终分数 → 入总账
        new_eps = finished_returns[n_before:]                        # 只看本轮新完成的那几局（切片从 n_before 开始）
        ep_returns = [ep_returns[-1]]                                # 只保留仍在进行中那一局的累计值，列表长度重置为 1
        stats, _ = ppo_update(model, optimizer, buf, cfg)            # 用这批数据做多 epoch 更新（数据用完即弃）
        mean_ret = float(np.mean(new_eps)) if new_eps else float("nan")   # 本轮平均回报；若 2048 步内一局都没结束则为 nan
        history.append((update, mean_ret, stats))                    # 存进历史，供画图
        if update % cfg.log_interval == 0:                           # 按间隔打印（默认每轮）
            print(                                                        # 一行日志看五个信号：
                f"[upd {update:4d}] ep_ret={mean_ret:7.1f}  "              #   1) 回报 ↑ 就是学起来了（CartPole 上限 500）
                f"pi_loss={stats['policy_loss']:+.3f}  v_loss={stats['value_loss']:.3f}  "  # 2)3) 策略/价值损失量级
                f"ent={stats['entropy']:.3f}  kl={stats['kl']:.4f}  clip={stats['clip_frac']*100:.1f}%"  # 4) 熵（探索度）5) KL 与截断比例（漂移程度）
            )
        if update % cfg.eval_interval == 0 or update == cfg.total_updates:   # 定期评测；最后一轮无论如何都评一次
            score = evaluate(model, eval_env, cfg)                   # 贪心策略跑 5 局取平均
            print(f"    -> eval greedy reward = {score:.1f}  (elapsed {time.time()-start:.1f}s)")   # 顺带打印累计耗时
            if score > best_eval:                                    # 只要刷新历史最好成绩
                best_eval = score                                    #   更新记录
                torch.save({"model": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, "ppo_best.pt")
                # ↑ 存权重 + 两个维度信息：以后加载模型时不必再问环境，直接按这两个数重建网络结构
            threshold = getattr(env.spec, "reward_threshold", None) or 475.0   # 环境的"通关线"（CartPole-v1 是 475）
            #   getattr 第三参 None 兜底：某些环境没有 reward_threshold 属性，就退回手写的 475
            if score >= threshold:                                   # 达标就停止训练
                print(f"Solved in {update} updates (threshold={threshold}).")
                break                                                # 提前结束，省时间
    env.close()                                                      # 关环境，释放渲染/子进程资源
    eval_env.close()                                                 # 评测环境同样要关
    return model, history, best_eval                                 # 返回模型、训练历史、最好分数

def plot_history(history, path):
    try:                                                             # matplotlib 是可选依赖：没装就优雅跳过，不因画图失败而让整个训练报错
        import matplotlib
        matplotlib.use("Agg")                                        # 用非交互后端：服务器/无显示器环境也能存图（不弹窗口）
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图：pip install matplotlib")
        return                                                       # 直接返回，不抛异常
    xs = [h[0] for h in history if not np.isnan(h[1])]                # 过滤掉 nan（那几轮没有局结束，没有回报可画）
    ys = [h[1] for h in history if not np.isnan(h[1])]                # 对应回报
    kl = [h[2]["kl"] for h in history if not np.isnan(h[1])]          # 对应平均 KL，注意筛选条件与上面一致，否则 x/y/kl 长度错位
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))                 # 1 行 2 列双子图：回报曲线 + KL 曲线
    axes[0].plot(xs, ys, color="#2a6fdb", linewidth=1.6)              # 左图：回报随 update 的变化（看学习是否发生）
    axes[0].set_title("episode return per update")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.25)                                          # 淡网格线，方便读数
    axes[1].plot(xs, kl, color="#d95f02", linewidth=1.6)              # 右图：KL 随 update 的变化（看策略是否暴走）
    axes[1].set_title("approx KL of policy update")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("KL")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()                                                # 自动调子图间距，防标题/标签重叠
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)          # 确保输出目录存在；`or "."` 处理"只写了文件名、没有目录"的情况（dirname 为空串）
    fig.savefig(path, dpi=160)                                        # 存图，160 dpi 保证清晰度
    print(f"learning curve saved to {path}")

def parse_args():
    p = argparse.ArgumentParser(description="PPO from scratch (single file)")   # 命令行解析器
    p.add_argument("--env-id", default=Config.env_id)                 # 每个超参都开一个命令行开关，默认值取自 Config
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--updates", type=int, default=Config.total_updates, dest="total_updates")
    # ↑ dest="total_updates"：命令行写 --updates，但存进命名空间的字段名叫 total_updates，
    #   这样后面 Config(**vars(args)) 才能和 Config 的字段名对齐
    p.add_argument("--rollout-steps", type=int, default=Config.rollout_steps)   # 连字符 → 下划线由 argparse 自动转换
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--lam", type=float, default=Config.lam)
    p.add_argument("--clip-eps", type=float, default=Config.clip_eps)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--minibatch-size", type=int, default=Config.minibatch_size)
    p.add_argument("--value-coef", type=float, default=Config.value_coef)
    p.add_argument("--entropy-coef", type=float, default=Config.entropy_coef)
    p.add_argument("--target-kl", type=float, default=Config.target_kl)
    p.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    p.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    p.add_argument("--log-interval", type=int, default=Config.log_interval)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", default="")                              # 画图路径，空串表示不画
    return p.parse_args()                                             # 解析并返回 Namespace 对象

def main():
    args = parse_args()                                               # 拿到命令行参数
    cfg = Config(**{k: v for k, v in vars(args).items() if not k == "plot"})
    # ↑ vars(args) 把 Namespace 转成 dict；字典推导过滤掉 plot（它不是 Config 的字段，硬传会 TypeError）
    #   ** 解包成关键字参数 → 用命令行值覆盖 Config 默认值。注意这里没做类型校验，传错类型会在训练中才炸
    model, history, best = train(cfg)                                 # 开始训练
    print(f"training finished. best greedy eval return = {best:.1f}")
    if args.plot:                                                     # 只有指定了路径才画图
        plot_history(history, args.plot)

if __name__ == "__main__":                                            # 只有"直接运行这个文件"才执行；被 import 时不自动训练
    main()
