"""
PPO for LLMs —— RLHF 中 PPO 的从零实现，单文件可运行。

和 CartPole 版 PPO 的区别不在 clip，而在"状态、动作、奖励从哪来"：
    CartPole       : 状态 = 4 维物理量，动作 = 左右推，奖励 = 环境每步给
    LLM (RLHF)     : 状态 = prompt + 已生成的 token 前缀
                     动作 = 下一个 token（词表大小 13）
                     奖励 = 奖励模型给整条回答打分（只有句末一个稀疏奖励）
                            + 每个 token 上对参考模型的 KL 惩罚

这也是 InstructGPT 那一套三步流程，本文件把三步都实现出来：
    ① SFT        : 监督微调，得到初始策略 π_SFT（同时作为参考模型 π_ref 冻结）
    ② Reward Model: Bradley–Terry 成对比较训练打分器 r_φ
    ③ PPO        : Actor（策略 + 价值头）用 token 级 GAE + 截断目标优化
                   reward_t = r_φ(x, y) · 1[t = 末位] − β · KL_t(π_θ ‖ π_ref)

与 GRPO 的关系：策略项骨架完全相同（token 级 ratio + clip）。
结构性差别只有一处 —— 优势 A 的来源：
    PPO  : Critic 估 V(s_t)，沿 token 时间轴做 GAE
    GRPO : 同一 prompt 采 G 条回答，用组内 Z-score 当优势，全轨迹共享

结构：
    1. Config             超参数
    2. ArithmeticTask     题库 + 规则校验（只用于评测/日志，训练奖励来自 RM）
    3. PolicyLM           Actor-Critic：字符级 GRU + 策略头 + 价值头
    4. RewardModel        Bradley–Terry 奖励模型
    5. sft_pretrain       第①步：监督微调
    6. train_reward_model 第②步：训练奖励模型
    7. rollout_llm        采样回答，记录 old_logp / value / ref_logp / RM 分
    8. compute_gae_token  token 级 GAE（奖励稀疏在句末，KL 逐 token 塑形）
    9. ppo_update         第③步：token 级 clip 目标 + 价值回归 + 熵正则
   10. train              三步串起来的训练主循环
   11. selftest           数值断言（不训练，直接验证公式）

运行：
    python ppo.py                        # 默认 300 轮 PPO，CPU 约 1 分钟
    python ppo.py --reward rule          # 消融：用可验证奖励替代奖励模型
    python ppo.py --iters 400 --plot assets/learning_curve.png
    python ppo.py --selftest             # 只跑数值断言
"""

# --------------------------------------------------------------------------------------
# 依赖：标准库 4 个 + numpy + torch（全部 CPU 可跑，不需要 transformers / gymnasium）
# --------------------------------------------------------------------------------------
import argparse      # 命令行参数解析：把 Config 的字段一个个暴露成 --xxx 开关
import copy          # 用来 deepcopy 出一份 SFT 权重当参考模型 π_ref
import os            # 只是给绘图创建输出目录用
import random        # Python 侧随机数：题库采样、构造错误答案（与 torch 的随机流相互独立）
import time          # 训练计时，日志里打 elapsed
from dataclasses import dataclass   # 用 @dataclass 让 Config 只写字段、自动生成 __init__

import numpy as np                      # 只用于评测时的 mean，以及在 CPU 上做数组统计
import torch                            # 全部张量运算
import torch.nn as nn                   # nn.Module / nn.GRU / nn.Embedding / nn.Linear
import torch.nn.functional as F         # F.logsigmoid：Bradley–Terry 的 log σ
from torch.distributions import Categorical   # 离散动作分布：采样 token + 给 log_prob

# --------------------------------------------------------------------------------------
# 词表：0-9 十个数字 + "+" + "=" + EOS，共 13 个 token
# 特意做得很小，让整个 RLHF 流程能在 CPU 上几十秒跑完，同时保留 token 级的所有结构
# --------------------------------------------------------------------------------------
DIGITS = list("0123456789")   # 0..9 依次对应 id 0..9，顺序即编码
PLUS, EQ, EOS = 10, 11, 12    # 加号、等号、句末符（EOS 既是终止符也是 padding 符）
VOCAB_SIZE = 13               # 词表大小 = 10 + 3


# --------------------------------------------------------------------------------------
# 1. 超参数
# --------------------------------------------------------------------------------------
@dataclass            # 自动生成 __init__(...)，字段默认值即默认超参
class Config:
    seed: int = 0     # 全局随机种子：Python / numpy / torch 三路都设，保证可复现

    # ---------- ① SFT ----------
    sft_steps: int = 300           # 监督微调步数（故意不训满，留出 RL 的提升空间）
    sft_batch: int = 64            # SFT 每步的样本数（都是 (a, b) 题目对）
    sft_lr: float = 5e-4           # SFT 学习率：比 PPO 阶段大，因为是从零学语言模型
    train_ratio: float = 0.6       # SFT 只见过 60% 的题目，剩下的靠 RL 泛化

    # ---------- ② Reward Model ----------
    rm_steps: int = 600            # 奖励模型训练步数
    rm_batch: int = 64             # 每步 32 组成对比较（正例 + 负例）
    rm_lr: float = 1e-3             # 奖励模型学习率（小模型 + 成对损失，可以大一些）

    # ---------- ③ PPO ----------
    iters: int = 300               # PPO 迭代轮数（每轮采样 → 打分 → GAE → 多 epoch 更新）
    batch_prompts: int = 64        # 每轮 prompt 数
    samples_per_prompt: int = 1    # 每个 prompt 采几条回答（PPO 通常 1 条即可，靠 Critic 估基线）
    max_new_tokens: int = 3        # 回答最大长度（最大答案 18 → 两位数字 + EOS）
    temperature: float = 1.0       # 采样温度；新旧策略必须用同一温度，ratio 才有意义
    gamma: float = 1.0             # LLM 里通常不折扣，整条回答算一个"回合"
    lam: float = 0.95              # GAE 的 lambda：越小越依赖 Critic，越大越依赖实际回报
    clip_eps: float = 0.2          # ratio 截断范围 [1-eps, 1+eps]
    kl_coef: float = 0.02          # β：对参考模型的逐 token KL 惩罚
    value_coef: float = 0.5        # 价值损失权重
    entropy_coef: float = 0.01     # 熵奖励权重
    epochs: int = 2                # 同一批数据重复训练轮数
    minibatch_size: int = 16       # 按「序列」切分（一个 minibatch 是 16 条回答，不是 16 个 token）
    lr: float = 1e-4               # PPO 阶段学习率：比 SFT 小一个量级，防止一步走崩
    max_grad_norm: float = 0.5     # 梯度裁剪阈值，PPO 的老惯例
    target_kl: float = 0.03        # 新旧策略 KL 超过该值提前结束本批更新
    reward_source: str = "hybrid"  # model | rule | hybrid（hybrid = RM 分 + 答对奖励）
    reward_norm: bool = True       # 对整个 batch 的结果奖励做标准化，防止尺度过大压过 KL 项

    # ---------- 评测与日志 ----------
    eval_interval: int = 50        # 每隔多少轮在全部 55 道题上做一次贪心评测
    log_interval: int = 30         # 每隔多少轮打一行训练日志
    device: str = "cpu"            # 默认 CPU；模型很小，GPU 也快不到哪去


# --------------------------------------------------------------------------------------
# 2. 任务：题库 + 规则校验（训练不再用它给奖励，只用来评测和打日志）
# --------------------------------------------------------------------------------------
class ArithmeticTask:
    """一位数加法题库。

    只保留 a+b <= max_sum 的题（默认 9，不进位，答案是一位数）：
    55 道题全部参与评测，其中 train_ratio 的部分用于 SFT，
    这样 SFT 之后模型在整体题库上一定留有提升空间 —— 正是 RLHF 想解决的泛化问题。
    """

    def __init__(self, max_sum: int = 9):
        # 笛卡尔积 + 过滤：0+0 … 9+0 / 0+9 …，只留和不进位的组合，正好 55 道
        self.pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= max_sum]   # 55 道不进位加法题

    def split(self, ratio: float, seed: int):
        # 独立一个 Random(seed)：保证「哪些题进 SFT」完全由种子决定，与训练中的随机流解耦
        rng = random.Random(seed)
        # 复制一份再打乱，避免 shuffle 原地改掉 self.pairs（评测还要用全量题库）
        shuffled = self.pairs[:]
        rng.shuffle(shuffled)
        # 取前 ratio 比例作为 SFT 训练集；max(1, ...) 防止 ratio 太小时取到 0 道题
        n = max(1, int(len(shuffled) * ratio))
        return shuffled[:n]                        # 前 n 道就是 SFT 训练集

    @staticmethod
    def encode_prompt(a: int, b: int):
        # prompt 的固定格式：a + b = ，共 4 个 token，这就是 LLM 的「初始状态 s_0」
        return [a, PLUS, b, EQ]

    @staticmethod
    def answer_ids(a: int, b: int):
        """标准答案的 token 序列：数字 + EOS。"""
        # 先把和转成字符串再逐字符取 int，天然支持多位数（"18" -> [1, 8]）
        return [int(c) for c in str(a + b)] + [EOS]

    @staticmethod
    def score(a_arr, b_arr, comp: torch.Tensor):
        """规则打分器：返回 (reward, exact, well_formed)，各是长度 N 的数组。

        reward = 1.0                              完全答对
               = 0.2 + 0.8 * 逐位匹配率          格式正确但数字不对
               = 0.0                             格式都不对（没吐 EOS / 吐了非数字）
        """
        n = comp.size(0)                                        # 本批回答条数
        rewards = np.zeros(n, dtype=np.float32)                 # 连续奖励（0~1）
        exact = np.zeros(n, dtype=np.float32)                   # 是否完全答对（0/1）
        well = np.zeros(n, dtype=np.float32)                    # 格式是否合法（0/1）
        for i in range(n):                                      # 逐条回答按规则判分，不需要梯度
            a_i, b_i = int(a_arr[i]), int(b_arr[i])             # 还原这道题的 a、b
            ids = comp[i].tolist()                              # 这条回答的 token 序列
            if EOS not in ids:                                  # 没吐出 EOS：长度被 max_new 截断，判格式失败
                continue                                        # 三个数组都保持 0
            body = ids[: ids.index(EOS)]                        # 只取 EOS 之前的答案正文
            if len(body) == 0 or any(t < 0 or t > 9 for t in body):   # 空答案或混进了非数字 token
                continue                                        # 同样算格式失败
            well[i] = 1.0                                       # 到这里格式合法：数字 + EOS
            s = "".join(str(t) for t in body)                   # 把 token 拼回字符串，方便逐位比较
            ans = str(a_i + b_i)                                # 标准答案字符串
            if s == ans:                                        # 完全一致
                rewards[i] = 1.0                                # 满分
                exact[i] = 1.0                                  # 记为 exact
            else:                                               # 格式对但数字错：给部分分，制造奖励梯度
                match = sum(1 for x, y in zip(s, ans) if x == y)        # 逐位相同的字符数
                rewards[i] = 0.2 + 0.8 * match / max(len(s), len(ans))  # 基础分 0.2 + 逐位匹配加权
        return rewards, exact, well            # 连续分 / 是否答对 / 格式是否合法


def completion_mask(comp: torch.Tensor):
    """第一个 EOS（含）之前为 1，之后为 0。"""
    is_eos = comp == EOS                                     # [B, L] 布尔：每个位置是不是 EOS
    has_eos = is_eos.any(1)                                  # [B] 这条回答有没有吐 EOS
    first = is_eos.float().argmax(1)                          # [B] 第一个 EOS 的下标（没有 EOS 时为 0，下面会修正）
    # 有 EOS → 长度 = 第一个 EOS 下标 + 1（EOS 本身也参与损失，因为它也是一个预测目标）
    # 没 EOS → 长度 = max_new_tokens，整条都要算
    length = torch.where(has_eos, first + 1, torch.full_like(first, comp.size(1)))
    pos = torch.arange(comp.size(1), device=comp.device).unsqueeze(0)   # [1, L] 位置下标
    return (pos < length.unsqueeze(1)).float()               # [B, L]：位置 < 长度 的为 1，即有效 token


# --------------------------------------------------------------------------------------
# 3. Actor-Critic：策略头 + 价值头共用一条 GRU 主干
# --------------------------------------------------------------------------------------
class PolicyLM(nn.Module):
    """字符级自回归语言模型，额外长了一个价值头（InstructGPT 里 actor 与 critic 合体）。

    - 策略头：输出下一个 token 的 logits → π_θ(a_t | s_t)
    - 价值头：输出当前前缀状态的 V(s_t) → 给 GAE 当基线

    关键点：价值头评估的是「已经生成完前 t 个 token 的这条前缀」，
    所以它和策略头取同一个隐状态，只是接不同的线性层。
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emb_dim: int = 48, hidden_dim: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)                  # token id → 48 维向量
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)      # 主干：沿 token 轴做循环编码
        self.head = nn.Linear(hidden_dim, vocab_size)                 # 策略头：隐状态 → 13 个 token 的 logits
        self.value_head = nn.Linear(hidden_dim, 1)                    # 价值头：隐状态 → 标量 V(s)
        # 策略头小增益：初始接近均匀分布，先探索；价值头常规增益
        nn.init.orthogonal_(self.head.weight, gain=0.01)              # 0.01 倍正交初始化 → 初始 logits 很平
        nn.init.constant_(self.head.bias, 0.0)                        # 偏置清零，保证初始分布无偏
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)         # 价值头正常初始化，别让 V 一开始就饱和
        nn.init.constant_(self.value_head.bias, 0.0)                  # V 初始约等于 0

    # ---------- 基础前向 ----------
    def encode_prompt(self, prompt_ids: torch.Tensor):
        """吃完整段 prompt，返回最后一步隐状态 (1, B, H)。"""
        _, h = self.gru(self.emb(prompt_ids))     # 整段 prompt 一次喂完；h 是最后时刻的隐状态
        return h                                  # 形状 (num_layers=1, B, H)，就是 s_0 的表示

    def step(self, token: torch.Tensor, h: torch.Tensor):
        """单步：吃一个 token，返回 (logits, 新隐状态)。"""
        # token 形状 [B]，补一个时间维变 [B, 1] 才能喂给 batch_first 的 GRU
        out, h = self.gru(self.emb(token).unsqueeze(1), h)
        return self.head(out.squeeze(1)), h       # out [B,1,H] → squeeze → [B,H] → logits [B,13]

    def head_logits(self, h: torch.Tensor):
        return self.head(h.squeeze(0))            # h [1,B,H] → [B,H] → logits

    def head_value(self, h: torch.Tensor):
        return self.value_head(h.squeeze(0)).squeeze(-1)   # [1,B,H] → [B,H] → [B,1] → [B]

    def logprobs(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, temperature: float = 1.0):
        """只算 log 概率，参考模型用（不必算价值）。"""
        # 参考模型永远冻结，只需要 logp / 熵，价值头的结果直接丢掉（少一次拷贝也无所谓）
        logp, ent, _ = self.logprobs_values(prompt_ids, comp_ids, temperature)   # 只要前两路输出
        return logp, ent                       # 参考模型永远不更新，不需要价值

    def logprobs_values(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, temperature: float = 1.0):
        """teacher forcing：一次前向算出 completion 每个位置的 log 概率、熵与 V(s_t)。

        整段序列 = prompt + completion；取第 P-1 位之后的输出，
        即用「prompt + 前 t 个 token」去预测第 t 个 token。
        """
        ids = torch.cat([prompt_ids, comp_ids], dim=1)          # 拼成 [B, P+L]
        out, _ = self.gru(self.emb(ids[:, :-1]))                # 去掉最后一个 token：它是"标签"不是"输入"
        # 前 P-1 个位置预测的是 prompt 内部的 token（不算 loss），从第 P-1 位起才是预测 completion
        out = out[:, prompt_ids.size(1) - 1 :, :]                     # [B, L, H]
        logits = self.head(out) / temperature                     # 温度缩放：和采样时的分布保持一致
        logp = torch.log_softmax(logits, dim=-1)                  # [B, L, V] 对数概率
        ent = -(logp.exp() * logp).sum(-1)                       # [B, L] 每步的熵
        picked = logp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)   # [B, L] 取实际生成的 token 的 logp
        values = self.value_head(out).squeeze(-1)                      # [B, L] V(s_t)：预测第 t 个 token前的状态价值
        return picked, ent, values

    # ---------- 采样 ----------
    @torch.no_grad()                                          # rollout 只采数据，不建图，省显存也省时间
    def sample(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """逐 token 自回归采样，同时记录 old_logp 和 V(s_t)。

        返回 comp / logp / values / mask，形状都是 [B, L]。
        """
        B = prompt_ids.size(0)                                            # batch 里有多少条回答
        h = self.encode_prompt(prompt_ids)                                # 先把 prompt 吃完，得到 s_0
        # 用 EOS 预填充：还没生成的位置默认就是 EOS，天然充当 padding
        comp = torch.full((B, max_new_tokens), EOS, dtype=torch.long, device=prompt_ids.device)
        logps = torch.zeros((B, max_new_tokens), dtype=torch.float32, device=prompt_ids.device)   # 每步采样 logp
        values = torch.zeros((B, max_new_tokens), dtype=torch.float32, device=prompt_ids.device)  # 每步 V(s_t)
        alive = torch.ones((B,), dtype=torch.float32, device=prompt_ids.device)   # 1=还没吐 EOS，0=已结束

        for t in range(max_new_tokens):                                   # 最多生成 L 步
            logits = self.head_logits(h)                  # 历史 token 已全部吃完，直接解码
            values[:, t] = self.head_value(h)             # V(s_t)：生成第 t 个 token前的状态价值
            dist = Categorical(logits=logits / temperature)               # 用 logits 构造离散分布（内部会 softmax）
            tok = dist.sample()                                           # 采样出一个 token [B]
            logps[:, t] = dist.log_prob(tok) * alive                      # 已结束的序列 logp 记为 0（掩码掉）
            # 已结束的序列这一位强制写 EOS，保证后面全是 EOS，mask 才能正确截断
            comp[:, t] = torch.where(alive.bool(), tok, torch.full_like(tok, EOS))
            alive = alive * (tok != EOS).float()                          # 刚吐了 EOS 的序列从下一步起置 0
            _, h = self.step(comp[:, t], h)                               # 把这一位喂回 GRU，推进状态 s_{t+1}

        return comp, logps, values, completion_mask(comp)                 # 四个都是 [B, L]

    @torch.no_grad()
    def greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int):
        """评测用贪心解码（等价于温度 → 0 的极限）。"""
        h = self.encode_prompt(prompt_ids)                                # 编码 prompt
        # 同样用 EOS 预填充；贪心解码不掩码，直接一路 argmax 到底
        comp = torch.full((prompt_ids.size(0), max_new_tokens), EOS, dtype=torch.long, device=prompt_ids.device)
        for t in range(max_new_tokens):
            logits = self.head_logits(h)                                  # 当前前缀 → logits
            comp[:, t] = logits.argmax(-1)                                # 取概率最大的 token
            _, h = self.step(comp[:, t], h)                               # 推进状态
        return comp                                                       # [B, L]，评测只看内容，不需要 logp/value


# --------------------------------------------------------------------------------------
# 4. 奖励模型：Bradley–Terry 成对比较
# --------------------------------------------------------------------------------------
class RewardModel(nn.Module):
    """给 (prompt, completion) 打一个标量分。

    结构就是一个小 GRU 编码器 + 线性头，取 EOS 位置的隐状态当整条回答的表示。
    训练目标是 Bradley–Terry：好回答的分数要比坏回答高
        L = -log σ(r(x, y_w) - r(x, y_l))
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emb_dim: int = 48, hidden_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)                  # 和策略模型同维度的 embedding
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)      # 奖励模型自己独立的 GRU，hidden 更大
        self.head = nn.Linear(hidden_dim, 1)                          # 输出 1 个标量分

    def forward(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor):
        ids = torch.cat([prompt_ids, comp_ids], dim=1)                # 拼成完整序列 [B, P+L]
        out, _ = self.gru(self.emb(ids))                              # 每个位置都有隐状态 [B, P+L, H]
        mask = completion_mask(comp_ids)                              # [B, L] 有效 token 掩码
        last = mask.sum(1).long().clamp(min=1) - 1     # EOS 所在位置（在 completion 内的相对下标）
        last = prompt_ids.size(1) + last                              # 加上 prompt 长度，换到整条序列的绝对下标
        # 按行号取「EOS 位置」的隐状态：一整条回答的语义就汇聚在最后一个有效 token 上
        pooled = out[torch.arange(out.size(0), device=out.device), last]   # 按行取 EOS 位置的隐状态
        return self.head(pooled).squeeze(-1)                          # [B] 每条回答一个分数


def bradley_terry_loss(r_win: torch.Tensor, r_lose: torch.Tensor):
    # -log σ(r_w - r_l)：胜者分数越高，σ 越接近 1，loss 越接近 0
    return -F.logsigmoid(r_win - r_lose).mean()


# --------------------------------------------------------------------------------------
# 5. 第①步：SFT —— 监督微调，得到初始策略
# --------------------------------------------------------------------------------------
def sft_pretrain(model: PolicyLM, task: ArithmeticTask, pairs, cfg: Config):
    """只在 completion 上算交叉熵（prompt 部分不参与 loss），这是 SFT 的标准做法。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.sft_lr)   # SFT 用普通 Adam，不用 PPO 的优化器
    rng = random.Random(cfg.seed + 100)                              # 独立随机流：+100 与别处错开
    model.train()                                                    # 打开 train 模式（虽然本模型没有 dropout/bn）
    for step in range(1, cfg.sft_steps + 1):
        batch = [rng.choice(pairs) for _ in range(cfg.sft_batch)]    # 有放回采样，简单起见不遍历全 epoch
        # prompt：每道题 4 个 token，[B, 4]
        prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in batch], device=cfg.device)
        # completion：答案 + EOS，再补 EOS 到 max_new_tokens 长度，保证 batch 内等长
        comp_ids = torch.tensor(
            [task.answer_ids(a, b) + [EOS] * (cfg.max_new_tokens - len(task.answer_ids(a, b)))
             for a, b in batch],
            device=cfg.device,
        )
        mask = completion_mask(comp_ids)                             # 只在 EOS 之前算 loss
        logp, _, _ = model.logprobs_values(prompt_ids, comp_ids, 1.0)  # teacher forcing 一次前向拿全部 logp
        loss = -((logp * mask).sum() / mask.sum())                    # 交叉熵：按有效 token 数归一化（不是按 batch）

        optimizer.zero_grad(set_to_none=True)                         # 清梯度；set_to_none 比置零省一次写
        loss.backward()                                               # 反传
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)   # 梯度裁剪，防爆
        optimizer.step()                                              # 更新

        if step % 40 == 0:                                            # 每 40 步打一次 loss
            print(f"[sft {step:4d}] ce_loss={loss.item():.4f}")
    model.eval()                                                      # 训练完切回 eval，供后面采样/评测
    return model


# --------------------------------------------------------------------------------------
# 6. 第②步：训练奖励模型
# --------------------------------------------------------------------------------------
def train_reward_model(rm: RewardModel, policy: PolicyLM, task: ArithmeticTask, pairs, cfg: Config):
    """在策略自己的输出分布上做比较，而不是只拿标准答案比构造错误。

    RLHF 里人来标注的就是「策略采样出来的这一批回答」，所以奖励模型必须见过
    模型自己会写出来的东西（包括答对的、答错的、格式崩坏的），否则 PPO 一旦
    偏离构造分布，就会精准地找到奖励模型的盲区 —— 奖励攻击就是这么发生的。
    """
    optimizer = torch.optim.Adam(rm.parameters(), lr=cfg.rm_lr)       # 奖励模型独立优化器
    rng = random.Random(cfg.seed + 200)                               # 独立随机流
    rm.train()                                                        # 训练模式

    for step in range(1, cfg.rm_steps + 1):
        batch = [rng.choice(pairs) for _ in range(cfg.rm_batch)]      # 每个 item 是 (a, b)
        prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in batch], device=cfg.device)   # [B,4]

        # ① 标准答案（永远正确）
        gold_ids = torch.tensor(
            [task.answer_ids(a, b) + [EOS] * (cfg.max_new_tokens - len(task.answer_ids(a, b)))
             for a, b in batch],
            device=cfg.device,
        )
        # ② 策略采样（贴近 PPO 真实的动作分布）：用当前策略跑一遍采样，只要 comp
        samp_ids, _, _, _ = policy.sample(prompt_ids, cfg.max_new_tokens, 1.0)
        # ③ 构造错误（覆盖格式崩坏 / 数字写错）
        wrong_ids = torch.empty_like(samp_ids)                        # 同形状、同 dtype
        for i, (a, b) in enumerate(batch):
            if rng.random() < 0.5:                                    # 一半概率：完全随机 token（造格式崩坏样本）
                wrong_ids[i] = torch.randint(0, 10, (cfg.max_new_tokens,), device=cfg.device)
            else:                                                     # 另一半：假的答案，格式仍然合法
                d = str((a + b + rng.choice([-1, 1, 2])) % 10)         # 和 ±1/2 再取模，保证是数字
                seq = [int(c) for c in d] + [EOS]                      # 转成 token 序列 + EOS
                seq = seq[: cfg.max_new_tokens] + [EOS] * max(0, cfg.max_new_tokens - len(seq))   # 补齐到定长
                wrong_ids[i] = torch.tensor(seq[: cfg.max_new_tokens], device=cfg.device)

        a_arr = torch.tensor([a for a, _ in batch])                   # 还原 a，供规则打分器判对错
        b_arr = torch.tensor([b for _, b in batch])                   # 还原 b
        _, samp_exact, samp_well = task.score(a_arr, b_arr, samp_ids)  # 采样回答里哪些其实是答对的
        samp_exact = torch.as_tensor(samp_exact)                      # numpy → tensor，供布尔索引

        r_gold = rm(prompt_ids, gold_ids)                             # [B] 标准答案的分数
        r_samp = rm(prompt_ids, samp_ids)                             # [B] 采样回答的分数
        r_wrong = rm(prompt_ids, wrong_ids)                           # [B] 错误回答的分数

        losses = [bradley_terry_loss(r_gold, r_wrong)]          # 正确答案 > 构造错误
        if samp_exact.sum() > 0:                                # 采样答对的也要被夸
            idx = samp_exact.bool()                             # 选出「采样恰好答对」的那些行
            losses.append(bradley_terry_loss(r_samp[idx], r_wrong[idx]))   # 它们应排在构造错误之前
        if (~samp_exact.bool()).sum() > 0:                      # 采样答错的要被压
            idx = ~samp_exact.bool()                            # 选出「采样答错」的那些行
            losses.append(bradley_terry_loss(r_gold[idx], r_samp[idx]))    # 标准答案应排在它们之前
        loss = torch.stack([l * 1.0 for l in losses]).mean()    # 三路比较等权平均（*1.0 只是显式写权重）

        optimizer.zero_grad(set_to_none=True)                          # 清梯度
        loss.backward()                                                # 反传
        nn.utils.clip_grad_norm_(rm.parameters(), cfg.max_grad_norm)   # 裁剪
        optimizer.step()                                               # 更新

        if step % 100 == 0:                                            # 每 100 步打一次健康度指标
            acc = (r_gold > r_wrong).float().mean().item()             # 排序准确率：gold 赢 wrong 的比例
            margin = (r_gold - r_wrong).mean().item()                  # 平均分差，看奖励是否"拉得开"
            print(f"[rm  {step:4d}] bt_loss={loss.item():.4f}  gold>wrong={acc*100:5.1f}%  margin={margin:+.3f}")
    rm.eval()                                                          # 训练完冻结语义（参数后面还会 requires_grad_(False)）
    return rm


# --------------------------------------------------------------------------------------
# 7. 第③步之采样：一次 rollout 收集 PPO 需要的全部字段
# --------------------------------------------------------------------------------------
@torch.no_grad()                                   # 采样阶段不需要梯度，所有记录都是"旧策略"的常数
def rollout_llm(policy: PolicyLM, ref_model: PolicyLM, rm: RewardModel,
                task: ArithmeticTask, pairs, cfg: Config):
    """对一批 prompt 采样回答，返回 batch 字典。

    token 级奖励的设计（对应 InstructGPT 的 reward shaping）：
        r_t = -β · (log π_θ(a_t|s_t) − log π_ref(a_t|s_t))   每个 token 都给
        r_T += r_φ(x, y)                                      只在句末给一次
    这样稀疏的整句奖励被 KL 塑形成逐 token 的稠密信号，Critic 才学得动。
    """
    S = cfg.samples_per_prompt                                           # 每个 prompt 采几条（PPO 默认 1）
    prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in pairs], device=cfg.device)   # [N,4]
    prompt_ids = prompt_ids.repeat_interleave(S, 0)                      # 每条 prompt 复制 S 份 → [N*S,4]

    # 采样：拿到 completion、旧策略 logp、旧价值、有效位置掩码
    comp, old_logp, values, mask = policy.sample(prompt_ids, cfg.max_new_tokens, cfg.temperature)
    # 参考模型的 logp：同一段 (prompt, completion) 用同一个温度重算，才能逐 token 比
    ref_logp, _ = ref_model.logprobs(prompt_ids, comp, cfg.temperature)

    # 规则校验（只用于日志 / 可选奖励）
    a_arr = torch.tensor([a for a, _ in pairs for _ in range(S)])        # 按 repeat_interleave 的顺序展开 a
    b_arr = torch.tensor([b for _, b in pairs for _ in range(S)])        # 同样展开 b
    rule_reward, exact, well = task.score(a_arr, b_arr, comp)            # 规则打分：连续分 / 答对 / 格式合法
    rule_reward = torch.as_tensor(rule_reward, device=cfg.device)        # numpy → tensor（CPU 上零拷贝）
    rm_reward = rm(prompt_ids, comp)                                     # 奖励模型给整条回答打分，[N*S]

    # KL：逐 token 的 log 比，k1 形式（可正可负），保持无偏
    kl_t = old_logp - ref_logp                                           # [B,L]：log π_θ - log π_ref，可正可负
    token_rewards = -cfg.kl_coef * kl_t * mask                           # 逐 token 惩罚；掩码外的位置记 0
    last_idx = (mask.sum(1).long() - 1).clamp(min=0)                     # 每条回答最后一个有效 token（通常是 EOS）
    rows = torch.arange(prompt_ids.size(0), device=cfg.device)           # 行下标，用来做二维索引

    if cfg.reward_source == "model":                                     # 纯 RLHF：只信奖励模型
        outcome = rm_reward
    elif cfg.reward_source == "rule":                                    # 消融/上界：用可验证奖励
        outcome = rule_reward
    else:  # hybrid：奖励模型分 + 答对奖励
        outcome = rm_reward + rule_reward                                 # 两者相加，兼顾"像人评"和"真答对"

    if cfg.reward_norm:
        # 只做平移缩放，不改变批内的相对排序；好处是 KL 项的量级能和奖励对齐
        outcome = (outcome - outcome.mean()) / (outcome.std(unbiased=False) + 1e-6)

    # 整句奖励只在最后一个有效 token 上加一次，其它位置只承担 KL —— 这就是 sparse outcome reward
    token_rewards[rows, last_idx] += outcome

    batch = {                                                            # 打包本轮所有字段，供 GAE 与更新使用
        "prompt_ids": prompt_ids,                                        # [B,P]
        "comp_ids": comp,                                                # [B,L]
        "old_logp": old_logp,                                            # [B,L] 旧策略 logp，算 ratio 的分母
        "values": values,                                                # [B,L] 旧 Critic 的 V(s_t)，GAE 的基线
        "mask": mask,                                                    # [B,L] 有效位置
        "token_rewards": token_rewards,                                  # [B,L] KL 塑形 + 句末 outcome
        "rm_reward": rm_reward,                                          # [B] 仅日志/训练奖励
        "rule_reward": rule_reward,                                      # [B] 仅日志/可选奖励
        "exact": exact,                                                  # [B] 规则判定的答对率
        "well": well,                                                    # [B] 规则判定的格式合法率
        "kl_t": kl_t,                                                    # [B,L] 逐 token KL，用于监控
    }
    return batch          # 交给下面算 GAE 与 PPO 更新


# --------------------------------------------------------------------------------------
# 8. token 级 GAE
# --------------------------------------------------------------------------------------
def compute_gae_token(rewards, values, mask, gamma, lam):
    r"""逐 token 倒序递推（掩码之后的位置不参与）：

        δ_t = r_t + γ V(s_{t+1}) · m_{t+1} − V(s_t)
        A_t = δ_t + γλ · m_{t+1} · A_{t+1}
        return_t = A_t + V(s_t)

    形状均为 [B, L]。
    """
    B, L = rewards.shape                                       # B 条回答，每条最多 L 个 token
    adv = torch.zeros_like(rewards)                            # 优势张量，先占位
    last = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)   # 递推量 A_{t+1}，从最后一位之后开始为 0

    for t in reversed(range(L)):                               # 时间轴倒序：先算靠后的 token
        # 注意 t+1 可能越界：最后一位的"下一个状态"按 0 处理，相当于终止
        next_mask = mask[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        next_value = values[:, t + 1] if t + 1 < L else torch.zeros(B, device=rewards.device)
        # TD 误差：奖励 + 折扣后的下一状态价值（只在下一 token 有效时才算）− 当前价值
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        # GAE 递推：当前 δ 加上 γλ 加权的后续优势；next_mask 保证不会跨过 EOS 往后串
        last = delta + gamma * lam * next_mask * last
        adv[:, t] = last * mask[:, t]                          # 无效位置（EOS 之后）优势记 0

    returns = adv + values                                     # return = A + V，作为 Critic 的回归目标
    return adv, returns       # 优势给策略损失，return 给价值回归


# --------------------------------------------------------------------------------------
# 9. PPO 更新：token 级截断目标
# --------------------------------------------------------------------------------------
def clip_objective(ratio: torch.Tensor, adv: torch.Tensor, eps: float):
    """min(ratio*A, clip(ratio)*A)，返回逐 token 值。"""
    surr1 = ratio * adv                                              # 未截断的替代目标
    surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv           # 把 ratio 掐在 [1-ε, 1+ε]
    # 取 min：A>0 时防止 ratio 涨太多，A<0 时防止 ratio 跌太多 —— 只在"改进方向"上允许比例变化
    return torch.min(surr1, surr2)


def ppo_update(policy: PolicyLM, optimizer, batch, advantages, returns, cfg: Config):
    prompt_ids, comp_ids = batch["prompt_ids"], batch["comp_ids"]    # 本轮固定的 prompt / 已采样的回答
    old_logp, mask = batch["old_logp"], batch["mask"]                # 旧策略 logp（分母）/ 有效位置
    n = prompt_ids.size(0)                                           # 本批序列条数（minibatch 按序列切）
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,  # 累计统计，最后除以更新次数
             "kl": 0.0, "clip_frac": 0.0, "n": 0}

    for _ in range(cfg.epochs):                                      # 同一批数据重复训 epochs 遍
        perm = torch.randperm(n, device=cfg.device)                  # 每遍重新打乱
        for start in range(0, n, cfg.minibatch_size):
            mb = perm[start: start + cfg.minibatch_size]             # 一个 minibatch 的行下标

            new_logp, ent, new_values = policy.logprobs_values(       # 当前策略重新前向
                prompt_ids[mb], comp_ids[mb], cfg.temperature
            )
            m = mask[mb]                                             # 这个 minibatch 的有效位置
            log_ratio = new_logp - old_logp[mb]                      # log(π_new/π_old)，先算 log 更稳
            ratio = log_ratio.exp()                                  # 重要性采样比
            adv = advantages[mb]                                     # token 级优势

            # 策略损失：逐 token 取 min，再按 token 数归一化
            obj = clip_objective(ratio, adv, cfg.clip_eps) * m       # 掩码掉 EOS 之后的位置
            policy_loss = -(obj.sum() / m.sum())                     # 取负做最小化；分母是 token 数不是序列数

            # 价值损失：回归 GAE 的 return
            value_loss = (((new_values - returns[mb]) ** 2) * m).sum() / m.sum()   # 掩码 MSE

            # 熵奖励：按 token 平均
            entropy = (ent * m).sum() / m.sum()                      # 越大越探索，所以下面要减

            # 总损失：策略 + c1·价值 − c2·熵
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)                    # 清梯度
            loss.backward()                                          # 反传
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)   # 裁剪
            optimizer.step()                                         # 一步更新

            # 监控量：approx_kl 用 k3 等价式的低方差近似 (r-1) - log r
            approx_kl = (((ratio - 1.0) - log_ratio) * m).sum() / m.sum()
            # clip_frac：被截断（|r-1| > ε）的 token 占比，健康时应远小于 100%
            clip_frac = (((ratio - 1.0).abs() > cfg.clip_eps).float() * m).sum() / m.sum()
            stats["policy_loss"] += policy_loss.item()               # 累加
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["kl"] += approx_kl.item()
            stats["clip_frac"] += clip_frac.item()
            stats["n"] += 1                                          # 计一次 minibatch 更新

        if stats["n"] and stats["kl"] / stats["n"] > cfg.target_kl:
            break  # 新旧策略偏离过大，提前停止本批更新

    for k in ("policy_loss", "value_loss", "entropy", "kl", "clip_frac"):   # 求平均
        stats[k] /= max(stats["n"], 1)                                # 防 0 除
    return stats          # 本轮 minibatch 平均指标


# --------------------------------------------------------------------------------------
# 10. 训练主循环：SFT → RM → PPO
# --------------------------------------------------------------------------------------
def train(cfg: Config):
    random.seed(cfg.seed)                                   # Python 侧种子
    np.random.seed(cfg.seed)                                # numpy 侧种子
    torch.manual_seed(cfg.seed)                             # torch 侧种子
    task = ArithmeticTask()                                 # 全量 55 道题（评测用）
    rng = random.Random(cfg.seed + 1)                       # 训练采样用的独立随机流（+1 与 SFT 的 +100 错开）
    sft_pairs = task.split(cfg.train_ratio, cfg.seed)       # 交给 SFT 的那 60%
    t0 = time.time()                                        # 计时起点

    # ① SFT：初始策略
    policy = PolicyLM().to(cfg.device)                      # 从随机初始化开始
    sft_pretrain(policy, task, sft_pairs, cfg)              # 只在 sft_pairs 上做监督学习（这一步直接改参数，不返回也生效）
    acc_before = greedy_accuracy(policy, task, task.pairs, cfg)   # 全量题库上的 SFT 基线
    print(f"[sft done] greedy exact on all {len(task.pairs)} problems = {acc_before*100:.1f}%")
    print(f"           (SFT 只见过 {len(sft_pairs)} 道题，其余靠 RL 泛化)")

    # 参考模型：SFT 权重的一份冻结副本，KL 就是往它身上拉
    ref_model = copy.deepcopy(policy).to(cfg.device)        # 深拷贝，之后 policy 更新不会影响它
    ref_model.eval()                                        # 推理模式
    for p in ref_model.parameters():                        # 逐参数关掉梯度
        p.requires_grad_(False)                          # 冻结参考模型参数

    # ② 奖励模型
    rm = RewardModel().to(cfg.device)                       # 独立初始化
    train_reward_model(rm, policy, task, task.pairs, cfg)   # 用全量题库采样来训奖励模型（RL 阶段会采到任意题）
    for p in rm.parameters():                               # 训练完冻结：PPO 阶段只当打分器
        p.requires_grad_(False)
    rm.eval()

    # ③ PPO
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)   # 只优化策略（含价值头）
    history = []                                            # 每轮一条字典，供画曲线

    for it in range(1, cfg.iters + 1):                  # 一轮 = 采样 → GAE → 多次 epoch 更新
        pairs = [rng.choice(task.pairs) for _ in range(cfg.batch_prompts)]   # 每轮随机抽 64 题
        batch = rollout_llm(policy, ref_model, rm, task, pairs, cfg)         # 采样 + 打分 + KL 塑形

        adv, returns = compute_gae_token(                    # token 级 GAE
            batch["token_rewards"], batch["values"], batch["mask"], cfg.gamma, cfg.lam
        )
        m = batch["mask"]                                    # 算 token 级均值/方差要排除 padding
        mean = (adv * m).sum() / m.sum()                     # 有效 token 上的优势均值
        std = (((adv - mean) ** 2) * m).sum() / m.sum()      # 有效 token 上的方差
        adv = (adv - mean) / (std.sqrt() + 1e-8)      # token 级标准化：稳定策略梯度尺度
        adv = adv * m                                         # 标准化后再乘掩码，padding 位置保持 0

        stats = ppo_update(policy, optimizer, batch, adv, returns, cfg)      # 多 epoch 的 clip 更新

        rm_mean = batch["rm_reward"].mean().item()           # 本批平均 RM 分
        rule_mean = batch["rule_reward"].mean().item()       # 本批平均规则分
        exact = float(np.mean(batch["exact"]))               # 采样回答里答对的占比
        well = float(np.mean(batch["well"]))                 # 采样回答里格式合法的占比
        kl = ((batch["kl_t"] * m).sum() / m.sum()).item()    # 逐 token KL 的均值
        length = m.sum(1).mean().item()                      # 平均回答长度

        history.append({                                     # 记录这一轮的所有指标
            "iter": it, "rm": rm_mean, "rule": rule_mean, "exact": exact,
            "well": well, "len": length, "sampled_kl": kl, **stats,   # 把 ppo_update 的统计量展开进来
        })

        if it % cfg.log_interval == 0:                       # 定期打训练日志
            print(
                f"[iter {it:4d}] rm={rm_mean:+.3f}  rule={rule_mean:.3f}  exact={exact*100:5.1f}%  "
                f"well={well*100:5.1f}%  kl={kl:.4f}  ent={stats['entropy']:.3f}  "
                f"clip={stats['clip_frac']*100:4.1f}%  v_loss={stats['value_loss']:.3f}"
            )

        if it % cfg.eval_interval == 0 or it == cfg.iters:   # 定期贪心评测（比采样更稳定）
            acc = greedy_accuracy(policy, task, task.pairs, cfg)
            print(f"    -> greedy exact on all {len(task.pairs)} problems = {acc*100:.1f}%  "
                  f"(elapsed {time.time()-t0:.1f}s)")

    return policy, rm, history, acc_before                   # 返回值给 main() 做收尾评测/画图


@torch.no_grad()
def greedy_accuracy(policy: PolicyLM, task: ArithmeticTask, pairs, cfg: Config):
    prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in pairs], device=cfg.device)   # 全部题的 prompt
    comp = policy.greedy(prompt_ids, cfg.max_new_tokens)             # 贪心解码，确定性强
    a_arr = torch.tensor([a for a, _ in pairs])                      # 还原 a / b
    b_arr = torch.tensor([b for _, b in pairs])
    _, exact, _ = task.score(a_arr, b_arr, comp)                     # 规则判分，只取 exact
    return float(np.mean(exact))                                     # 平均答对率


def plot_history(history, path):
    try:
        import matplotlib                                            # 延迟导入：不画图的人不必装
        matplotlib.use("Agg")                                        # 无窗口后端，服务器/容器里也能存图
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图：pip install matplotlib")
        return

    xs = [h["iter"] for h in history]                                # x 轴：PPO 轮数
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))                # 三联图

    axes[0].plot(xs, [h["exact"] * 100 for h in history], color="#2a6fdb", linewidth=1.6, label="exact (sampled)")
    axes[0].plot(xs, [h["well"] * 100 for h in history], color="#8da0cb", linewidth=1.2, linestyle="--", label="well-formed")
    axes[0].set_title("answer accuracy (sampled responses)")         # 图 1：采样回答的准确率与格式合法率
    axes[0].set_xlabel("PPO iteration")
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, [h["rm"] for h in history], color="#1b9e77", linewidth=1.6, label="reward model score")
    axes[1].plot(xs, [h["rule"] for h in history], color="#7570b3", linewidth=1.2, label="rule reward")
    axes[1].set_title("reward signal")                                # 图 2：两种奖励信号随训练的变化
    axes[1].set_xlabel("PPO iteration")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(xs, [h["sampled_kl"] for h in history], color="#d95f02", linewidth=1.6)
    axes[2].set_title("KL(π_θ ‖ π_ref) per token")                   # 图 3：逐 token KL，看策略漂移多远
    axes[2].set_xlabel("PPO iteration")
    axes[2].set_ylabel("KL")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()                                               # 自动调间距
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)         # 目录不存在就建（path 无目录时退化为 "."）
    fig.savefig(path, dpi=160)                                       # 存图
    print(f"learning curve saved to {path}")


# --------------------------------------------------------------------------------------
# 11. 数值断言：把公式和实现对齐
# --------------------------------------------------------------------------------------
def selftest():
    ok = []                                        # 收集通过的检查名，最后统一打印

    # ① token 级 GAE：两手算
    values = torch.tensor([[0.5, 0.6, 0.7]])       # 三个状态的价值，故意取简单数
    rewards = torch.tensor([[0.0, 0.0, 1.0]])      # 只有最后一步有奖励（模拟句末 outcome）
    mask = torch.ones(1, 3)                        # 三位全有效
    adv, ret = compute_gae_token(rewards, values, mask, gamma=1.0, lam=0.0)   # lam=0 → 退化成单步 TD
    # lam=0：A_t = r_t + V_{t+1} - V_t
    assert abs(adv[0, 2] - (1.0 - 0.7)) < 1e-6     # 最后一位：1.0 + 0(终止) - 0.7
    assert abs(adv[0, 1] - (0.0 + 0.7 - 0.6)) < 1e-6
    assert abs(adv[0, 0] - (0.0 + 0.6 - 0.5)) < 1e-6
    assert torch.allclose(ret, adv + values)       # return = A + V 的定义
    ok.append("token-level GAE matches hand calculation (lam=0)")

    # ② lam=1、gamma=1 时优势等于回报减去基线：整条轨迹的 reward 之和
    adv2, _ = compute_gae_token(torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1, 3),
                                torch.ones(1, 3), 1.0, 1.0)
    assert torch.allclose(adv2, torch.tensor([[1.0, 1.0, 1.0]]))   # V=0、λ=1 → 每个位置看到同样的 1.0
    ok.append("with gamma=lam=1 and V=0, advantage equals tail reward sum")

    # ③ 掩码之后的位置不贡献优势
    adv3, _ = compute_gae_token(torch.tensor([[0.0, 1.0, 9.0]]), torch.zeros(1, 3),
                                torch.tensor([[1.0, 1.0, 0.0]]), 1.0, 1.0)   # 第 2 位被掩掉
    assert adv3[0, 2].item() == 0.0                # 掩码位置的优势必须为 0，且不能把 9.0 往回串
    ok.append("masked positions carry zero advantage")

    # ④ ratio=1 时截断目标退化为 A，且 d(obj)/d(ratio) = A
    a = torch.tensor([1.0])                        # 优势取 1，符号好判断
    ratio = torch.ones(1, requires_grad=True)      # 比值正好落在 1（未被截断区）
    clip_objective(ratio, a, 0.2).backward()
    assert torch.allclose(ratio.grad, a)           # 在信任域内，梯度等于 A（会推动 ratio 上升）
    ratio = torch.tensor([1.5], requires_grad=True)   # 1.5 已超出 1.2，clip 生效
    clip_objective(ratio, a, 0.2).backward()
    assert ratio.grad.item() == 0.0                # A>0 且 ratio 过大 → 梯度为 0，不再鼓励继续涨
    ok.append("clip objective gradient = A at ratio=1, 0 outside the trust region")

    # ⑤ KL(k3) 恒非负
    torch.manual_seed(0)
    ref = torch.log_softmax(torch.randn(32, 5), -1)   # 造两组随机的对数概率
    new = torch.log_softmax(torch.randn(32, 5), -1)
    d = ref - new                                  # d = log q - log p（k3 的标准写法）
    k3 = d.exp() - d - 1.0                         # e^d - d - 1 ≥ 0 恒成立
    assert k3.min().item() >= -1e-6                # 数值误差允许 -1e-6
    ok.append("KL k3 estimator is non-negative")

    # ⑥ 采样记录的 logp 与 teacher forcing 重算的一致，value 形状对得上
    torch.manual_seed(0)
    m = PolicyLM()
    prompts = torch.tensor([[3, PLUS, 9, EQ], [4, PLUS, 5, EQ]])     # 两道题
    comp, logp_s, val_s, mask_s = m.sample(prompts, 3, temperature=1.0)
    logp_r, _, val_r = m.logprobs_values(prompts, comp, 1.0)          # 用同一段序列重算
    # 只比有效位置：padding 位置的采样 logp 被 alive 置 0，重算的也有值，不能直接全比
    assert torch.allclose(logp_s * mask_s, logp_r * mask_s, atol=1e-5)
    assert torch.allclose(val_s * mask_s, val_r * mask_s, atol=1e-5)  # 价值同样只在有效位上一致
    ok.append("sampling logp/value == teacher-forcing recompute")

    # ⑦ 奖励塑形：句末 token 拿到 RM 分，其余 token 只承担 KL 惩罚
    cfg = Config(batch_prompts=2, samples_per_prompt=1, max_new_tokens=3, kl_coef=0.02,
                 reward_source="model", reward_norm=False)           # 关掉标准化，方便手算期望
    task = ArithmeticTask()
    rm = RewardModel()                                               # 随机初始化的奖励模型，只验证"落点"不验证数值
    batch = rollout_llm(m, m, rm, task, [(3, 9), (4, 5)], cfg)       # policy 和 ref 用同一个模型 → KL 应为 0
    last = (batch["mask"].sum(1).long() - 1)                         # 每条回答的最后一个有效位
    rows = torch.arange(2)
    shaped = batch["token_rewards"][rows, last]                      # 取这些位置的奖励
    expected = batch["rm_reward"] - cfg.kl_coef * batch["kl_t"][rows, last]   # 期望 = RM 分 + 该位的 KL 惩罚
    assert torch.allclose(shaped, expected, atol=1e-5)
    ok.append("outcome reward lands on the last token, KL shaping on every token")

    # ⑧ Bradley–Terry：把胜者分数抬高，loss 必须下降
    r_win, r_lose = torch.tensor([0.5]), torch.tensor([0.0])         # 胜者 0.5，败者 0
    assert bradley_terry_loss(r_win, r_lose).item() < bradley_terry_loss(r_lose, r_win).item()   # 方向性检查
    ok.append("Bradley-Terry loss prefers higher score for the winner")

    for i, s in enumerate(ok, 1):                  # 逐条打印通过的检查
        print(f"  [{i}] OK  {s}")
    print(f"selftest passed ({len(ok)} checks)")


# --------------------------------------------------------------------------------------
# 12. 命令行入口
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PPO for LLMs / RLHF from scratch (single file)")
    p.add_argument("--seed", type=int, default=Config.seed)                            # 随机种子
    p.add_argument("--sft-steps", type=int, default=Config.sft_steps)                  # SFT 步数
    p.add_argument("--train-ratio", type=float, default=Config.train_ratio)            # 进 SFT 的题目比例
    p.add_argument("--rm-steps", type=int, default=Config.rm_steps)                    # 奖励模型步数
    p.add_argument("--iters", type=int, default=Config.iters)                          # PPO 轮数
    p.add_argument("--batch-prompts", type=int, default=Config.batch_prompts)          # 每轮 prompt 数
    p.add_argument("--samples-per-prompt", type=int, default=Config.samples_per_prompt)  # 每 prompt 采样条数
    p.add_argument("--temperature", type=float, default=Config.temperature)            # 采样温度
    p.add_argument("--gamma", type=float, default=Config.gamma)                        # 折扣因子
    p.add_argument("--lam", type=float, default=Config.lam)                            # GAE lambda
    p.add_argument("--clip-eps", type=float, default=Config.clip_eps)                  # ratio 截断宽度
    p.add_argument("--kl-coef", type=float, default=Config.kl_coef)                    # β：KL 惩罚系数
    p.add_argument("--value-coef", type=float, default=Config.value_coef)              # 价值损失权重
    p.add_argument("--entropy-coef", type=float, default=Config.entropy_coef)          # 熵奖励权重
    p.add_argument("--epochs", type=int, default=Config.epochs)                        # 每批数据训几遍
    p.add_argument("--minibatch-size", type=int, default=Config.minibatch_size)        # minibatch 序列数
    p.add_argument("--lr", type=float, default=Config.lr)                              # PPO 学习率
    p.add_argument("--target-kl", type=float, default=Config.target_kl)                # KL 早停阈值
    p.add_argument("--reward", choices=["model", "rule", "hybrid"], default=Config.reward_source,
                   help="训练奖励来源：奖励模型 / 可验证规则 / 两者相加")
    p.add_argument("--no-reward-norm", action="store_true", help="关闭结果奖励标准化（调试用）")
    p.add_argument("--eval-interval", type=int, default=Config.eval_interval)           # 贪心评测间隔
    p.add_argument("--log-interval", type=int, default=Config.log_interval)             # 日志间隔
    p.add_argument("--device", default="cpu")                                          # 设备
    p.add_argument("--plot", default="")                                               # 给定路径则画曲线
    p.add_argument("--selftest", action="store_true")                                  # 只跑数值断言
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:                                # 自检模式：不训练，直接退出
        selftest()
        return

    # 把 CLI 参数整体转成 Config：先排除不是 Config 字段的那几个（plot/selftest/开关名）
    kw = {k: v for k, v in vars(args).items() if k not in ("plot", "selftest", "reward", "no_reward_norm")}
    kw["reward_source"] = args.reward                # --reward 映射成 reward_source
    kw["reward_norm"] = not args.no_reward_norm      # --no-reward-norm 是反向开关
    cfg = Config(**kw)                               # 构造最终配置

    policy, rm, history, acc_before = train(cfg)     # 跑完 SFT → RM → PPO
    # 收尾再评测一次：train() 里的最后一次评测可能因 iters 不整除间隔而漏掉
    acc_after = greedy_accuracy(policy, ArithmeticTask(), ArithmeticTask().pairs, cfg)   # 全量题库上再评一次
    print(f"final: greedy exact {acc_before*100:.1f}% → {acc_after*100:.1f}%  (baseline SFT → after PPO)")
    if args.plot:                                    # 指定了 --plot 才画图
        plot_history(history, args.plot)


if __name__ == "__main__":                           # 既可作为脚本运行，也可被 import 复用（如 GRPO 对照）
    main()
