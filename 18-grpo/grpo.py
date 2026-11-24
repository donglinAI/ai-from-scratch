"""
GRPO (Group Relative Policy Optimization) —— PyTorch 从零实现，单文件可运行。

场景：一位数加法的「可验证任务」，不需要奖励模型，也不需要 Critic
    prompt      = "3+9="
    completion  = 策略网络逐个 token 生成，形如 "12<EOS>"
    reward      = 规则打分器：格式分 + 逐位匹配分

与 PPO 的关系：策略项骨架完全相同（token 级 ratio + clip）。
结构性差别只有一处 —— 优势 A 的来源：
    PPO  : Critic 估 V(s)，再沿时间做 GAE
    GRPO : 同一 prompt 采 G 条回答，用组内 Z-score 当优势，全轨迹共享

结构：
    1. Config             超参数
    2. ArithmeticTask     题目采样 + 可验证规则奖励
    3. PolicyLM           字符级 GRU 语言模型（Embedding + GRU + 输出头）
    4. sample_group       同一 prompt 采 G 条 completion
    5. group_advantages   组内 Z-score（GRPO 的核心）
    6. grpo_update        token 级 clip 目标 + KL(k3) + 熵
    7. train              采样 → 打分 → 组内归一化 → 多 epoch 更新
    8. selftest           数值断言（不训练，直接验证公式）

运行：
    python grpo.py                        # 默认 300 轮迭代，CPU 约 1~2 分钟
    python grpo.py --iters 600 --plot assets/learning_curve.png
    python grpo.py --selftest             # 只跑数值断言
"""

import argparse
import copy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# --------------------------------------------------------------------------------------
# 词表：0-9 十个数字 + "+" + "=" + EOS，共 13 个 token
# --------------------------------------------------------------------------------------
DIGITS = list("0123456789")
PLUS, EQ, EOS = 10, 11, 12
VOCAB_SIZE = 13


# --------------------------------------------------------------------------------------
# 1. 超参数
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 0
    iters: int = 300               # 迭代轮数（每轮：采一批组 → 打分 → 组内归一化 → 多 epoch 更新）
    batch_prompts: int = 16        # 每轮多少个 prompt
    group_size: int = 8            # G：每个 prompt 采多少条回答（GRPO 的分组大小）
    max_new_tokens: int = 3        # completion 最大长度（最大答案 18，两位数字 + EOS）
    temperature: float = 1.0       # 采样温度；旧/新策略的 log 概率用同一个温度，保持一致
    epochs: int = 2                # 同一批数据重复训练轮数
    minibatch_size: int = 32       # 按「序列」切分的 minibatch
    lr: float = 5e-4
    clip_eps_low: float = 0.2      # 比值下限压低幅度（DAPO 的 clip-higher 就是把它和上限拆开）
    clip_eps_high: float = 0.2     # 比值上限抬升幅度
    kl_coef: float = 0.02          # KL 惩罚权重 β（对参考模型）
    entropy_coef: float = 0.02     # 熵奖励权重
    max_grad_norm: float = 0.5
    normalize_std: bool = True     # 组内是否除以标准差（Dr. GRPO 建议关掉）
    drop_dead_groups: bool = False # 丢掉「全对 / 全错」的组（DAPO 的 dynamic sampling 思路）
    eval_interval: int = 20
    log_interval: int = 10
    device: str = "cpu"


# --------------------------------------------------------------------------------------
# 2. 任务与规则奖励：GRPO 只要求「能给一条回答打一个分」，不需要可微奖励
# --------------------------------------------------------------------------------------
class ArithmeticTask:
    """一位数加法题库 + 可验证奖励。

    只保留 a+b <= max_sum 的题目（默认 9，即不进位加法、答案是一位数），
    这样单 token 就能给出完整答案，玩具规模下 GRPO 的组内信号足够密。
    55 道题全部用于训练；评测用贪心解码跑同一批题，考的是「学会了没」。
    """

    def __init__(self, max_sum: int = 9, seed: int = 0):
        self.pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= max_sum]

    def sample_prompts(self, n: int, rng: random.Random):
        return [rng.choice(self.pairs) for _ in range(n)]

    @staticmethod
    def encode_prompt(a: int, b: int):
        return [a, PLUS, b, EQ]

    @staticmethod
    def score(a: int, b: int, comp: torch.Tensor, mask: torch.Tensor):
        """规则打分器：返回 (reward, exact, well_formed)，各是长度 N 的数组。

        reward = 1.0                              完全答对
               = 0.2 + 0.8 * 逐位匹配率          格式正确但数字不对
               = 0.0                             格式都不对（没吐 EOS / 吐了非数字）
        """
        n = comp.size(0)
        rewards = np.zeros(n, dtype=np.float32)
        exact = np.zeros(n, dtype=np.float32)
        well = np.zeros(n, dtype=np.float32)

        for i in range(n):
            a_i, b_i = int(a[i]), int(b[i])
            ids = comp[i].tolist()
            if EOS not in ids:                     # 没有 EOS：格式不合法
                continue
            body = ids[: ids.index(EOS)]
            if len(body) == 0 or any(t < 0 or t > 9 for t in body):
                continue
            well[i] = 1.0
            s = "".join(str(t) for t in body)
            ans = str(a_i + b_i)
            if s == ans:
                rewards[i] = 1.0
                exact[i] = 1.0
            else:
                match = sum(1 for x, y in zip(s, ans) if x == y)
                rewards[i] = 0.2 + 0.8 * match / max(len(s), len(ans))
        return rewards, exact, well


# --------------------------------------------------------------------------------------
# 3. 策略网络：字符级 GRU 语言模型
# --------------------------------------------------------------------------------------
class PolicyLM(nn.Module):
    """Embedding → GRU → 输出头。

    它就是一个极小的自回归语言模型：输入已生成的 token 序列，
    输出下一个 token 的 logits。GRPO 训练的是它的参数 θ。
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emb_dim: int = 48, hidden_dim: int = 96):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)
        # 输出头小增益初始化：初始策略接近均匀分布，先探索再收敛
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    # ---------- 基础前向 ----------
    def encode_prompt(self, prompt_ids: torch.Tensor):
        """吃完整段 prompt，返回最后一步的隐状态 (1, B, H)。"""
        out, h = self.gru(self.emb(prompt_ids))
        return h

    def step(self, token: torch.Tensor, h: torch.Tensor):
        """单步：吃一个 token，返回 (logits, 新隐状态)。"""
        out, h = self.gru(self.emb(token).unsqueeze(1), h)
        return self.head(out.squeeze(1)), h

    def head_logits(self, h: torch.Tensor):
        """只解码：把当前隐状态（已吃完历史 token）映射成下一个 token 的 logits。"""
        return self.head(h.squeeze(0))

    def logprobs(self, prompt_ids: torch.Tensor, comp_ids: torch.Tensor, temperature: float = 1.0):
        """teacher forcing：一次前向算出 completion 每个位置的 log 概率与熵。

        整段序列 = prompt + completion，取第 P-1 位之后的 logits 去预测 completion。
        """
        ids = torch.cat([prompt_ids, comp_ids], dim=1)
        logits, _ = self.gru(self.emb(ids[:, :-1]))
        logits = self.head(logits)[:, prompt_ids.size(1) - 1 :, :]   # [B, L, V]
        logits = logits / temperature
        logp = torch.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)                            # [B, L]
        picked = logp.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)   # [B, L]
        return picked, ent

    # ---------- 采样 ----------
    @torch.no_grad()
    def sample(self, prompt_ids: torch.Tensor, G: int, max_new_tokens: int, temperature: float = 1.0):
        """同一 prompt 采样 G 条 completion（逐 token 自回归）。

        返回 comp / old_logp / mask，三者形状都是 [B*G, L]。
        mask 覆盖到第一个 EOS（含）为止，EOS 之后的 token 不参与损失。
        """
        B, _ = prompt_ids.shape
        prompt = prompt_ids.repeat_interleave(G, 0)
        h = self.encode_prompt(prompt)

        comp = torch.full((B * G, max_new_tokens), EOS, dtype=torch.long, device=prompt.device)
        logps = torch.zeros((B * G, max_new_tokens), dtype=torch.float32, device=prompt.device)
        alive = torch.ones((B * G,), dtype=torch.float32, device=prompt.device)

        for t in range(max_new_tokens):
            logits = self.head_logits(h)                  # 历史 token 已全部吃完，直接解码
            dist = Categorical(logits=logits / temperature)
            tok = dist.sample()
            logps[:, t] = dist.log_prob(tok) * alive
            comp[:, t] = torch.where(alive.bool(), tok, torch.full_like(tok, EOS))
            alive = alive * (tok != EOS).float()
            _, h = self.step(comp[:, t], h)               # 用「实际写入序列的 token」推进状态

        mask = completion_mask(comp)
        return comp, logps, mask

    @torch.no_grad()
    def greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int):
        """评测用贪心解码（等价于温度 → 0 的极限）。"""
        h = self.encode_prompt(prompt_ids)
        comp = torch.full((prompt_ids.size(0), max_new_tokens), EOS, dtype=torch.long, device=prompt_ids.device)
        for t in range(max_new_tokens):
            logits = self.head_logits(h)              # prompt 已吃完，直接解码
            comp[:, t] = logits.argmax(-1)
            _, h = self.step(comp[:, t], h)
        return comp


def completion_mask(comp: torch.Tensor):
    """第一个 EOS（含）之前为 1，之后为 0。"""
    is_eos = comp == EOS
    has_eos = is_eos.any(1)
    first = is_eos.float().argmax(1)
    length = torch.where(has_eos, first + 1, torch.full_like(first, comp.size(1)))
    pos = torch.arange(comp.size(1), device=comp.device).unsqueeze(0)
    return (pos < length.unsqueeze(1)).float()


# --------------------------------------------------------------------------------------
# 4. 组内优势：GRPO 的全部魔法都在这十行里
# --------------------------------------------------------------------------------------
def group_advantages(rewards: torch.Tensor, G: int, eps: float = 1e-4, normalize_std: bool = True):
    """把每条回答的标量奖励变成组内相对优势。

        A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

    返回 (advantage, dead_group_ratio)。dead group = 组内奖励全相同，优势恒为 0。
    """
    r = rewards.view(-1, G)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, unbiased=False, keepdim=True)
    adv = (r - mean) / (std + eps) if normalize_std else (r - mean)
    dead = (std < 1e-6)
    adv = torch.where(dead, torch.zeros_like(adv), adv)   # 死组显式置零，避免 0/eps 的数值残渣
    return adv.reshape(-1), dead.float().mean().item()


# --------------------------------------------------------------------------------------
# 5. GRPO 更新
# --------------------------------------------------------------------------------------
def clip_objective(ratio: torch.Tensor, adv: torch.Tensor, eps_low: float, eps_high: float):
    """PPO/GRPO 共用的截断目标：min(ratio*A, clip(ratio)*A)，返回逐 token 值。"""
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    return torch.min(surr1, surr2)


def grpo_update(model: PolicyLM, ref_model: PolicyLM, optimizer, batch, cfg: Config):
    prompt_ids, comp_ids, old_logp, mask, adv = batch
    n = prompt_ids.size(0)
    stats = {"loss": 0.0, "kl": 0.0, "entropy": 0.0, "clip_frac": 0.0, "n": 0}

    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.minibatch_size):
            mb = perm[start : start + cfg.minibatch_size]
            p, c, m = prompt_ids[mb], comp_ids[mb], mask[mb]

            # 当前策略在整段 completion 上的 log 概率（teacher forcing，一次前向）
            new_logp, ent = model.logprobs(p, c, cfg.temperature)
            with torch.no_grad():
                ref_logp, _ = ref_model.logprobs(p, c, cfg.temperature)

            log_ratio = new_logp - old_logp[mb]
            ratio = log_ratio.exp()
            a = adv[mb].unsqueeze(1)                      # 整条回答共享一个优势

            obj = clip_objective(ratio, a, cfg.clip_eps_low, cfg.clip_eps_high) * m
            seq_obj = obj.sum(1) / m.sum(1)               # 1/|o_i| 长度归一化

            # KL(k3)：exp(ref-new) - (ref-new) - 1，恒非负，是 KL 的无偏估计
            d = ref_logp - new_logp
            kl = (d.exp() - d - 1.0) * m
            seq_kl = kl.sum(1) / m.sum(1)

            policy_loss = -seq_obj.mean()
            kl_loss = cfg.kl_coef * seq_kl.mean()
            ent_bonus = cfg.entropy_coef * (ent * m).sum() / m.sum()
            loss = policy_loss + kl_loss - ent_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            clip_frac = (((ratio - 1.0).abs() > cfg.clip_eps_high).float() * m).sum() / m.sum()
            stats["loss"] += loss.item()
            stats["kl"] += seq_kl.mean().item()
            stats["entropy"] += ent_bonus.item() / max(cfg.entropy_coef, 1e-12)
            stats["clip_frac"] += clip_frac.item()
            stats["n"] += 1

    for k in ("loss", "kl", "entropy", "clip_frac"):
        stats[k] /= max(stats["n"], 1)
    return stats


# --------------------------------------------------------------------------------------
# 6. 训练主循环
# --------------------------------------------------------------------------------------
def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    task = ArithmeticTask(seed=cfg.seed)
    rng = random.Random(cfg.seed + 1)
    model = PolicyLM().to(cfg.device)
    ref_model = copy.deepcopy(model).to(cfg.device)     # 参考策略：初始化后永久冻结
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    t0 = time.time()

    for it in range(1, cfg.iters + 1):
        pairs = task.sample_prompts(cfg.batch_prompts, rng)
        prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in pairs], device=cfg.device)

        # ① 采样：每个 prompt 采 G 条
        comp, old_logp, mask = model.sample(prompt_ids, cfg.group_size, cfg.max_new_tokens, cfg.temperature)
        prompt_seq = prompt_ids.repeat_interleave(cfg.group_size, 0)   # 与采样顺序对齐：prompt 主序、G 次序

        # ② 打分：规则验证器，不需要奖励模型
        rep_pairs = [pairs[i] for i in range(len(pairs)) for _ in range(cfg.group_size)]
        a_arr = torch.tensor([a for a, _ in rep_pairs])
        b_arr = torch.tensor([b for _, b in rep_pairs])
        rewards, exact, well = task.score(a_arr, b_arr, comp, mask)
        rewards_t = torch.as_tensor(rewards, device=cfg.device)

        # ③ 组内归一化：优势 = 组内 Z-score
        adv, dead_ratio = group_advantages(rewards_t, cfg.group_size, normalize_std=cfg.normalize_std)

        # 可选：跳过全对 / 全错的组（DAPO dynamic sampling 的最小版本）
        if cfg.drop_dead_groups:
            r = rewards_t.view(-1, cfg.group_size)
            keep = (r.std(dim=1, unbiased=False) > 1e-6)
            keep_seq = keep.repeat_interleave(cfg.group_size)
            prompt_seq, comp, old_logp, mask = prompt_seq[keep_seq], comp[keep_seq], old_logp[keep_seq], mask[keep_seq]
            adv = adv[keep_seq]
            if adv.numel() == 0:          # 整批都是死组：这一轮没有可学的信号，直接跳过
                continue

        batch = (prompt_seq, comp, old_logp, mask, adv)
        stats = grpo_update(model, ref_model, optimizer, batch, cfg)

        # 统计（按组看，比按序列看更能反映「这道题学没学会」）
        g_exact = torch.as_tensor(exact).view(-1, cfg.group_size).mean(1).mean().item()
        g_well = torch.as_tensor(well).view(-1, cfg.group_size).mean(1).mean().item()
        g_reward = rewards_t.view(-1, cfg.group_size).mean().item()
        mean_len = mask.sum(1).mean().item()

        history.append(
            {
                "iter": it,
                "exact": g_exact,
                "well": g_well,
                "reward": g_reward,
                "dead": dead_ratio,
                "len": mean_len,
                **stats,
            }
        )

        if it % cfg.log_interval == 0:
            print(
                f"[iter {it:4d}] reward={g_reward:.3f}  exact={g_exact*100:5.1f}%  well={g_well*100:5.1f}%  "
                f"dead={dead_ratio*100:4.1f}%  len={mean_len:.2f}  kl={stats['kl']:.4f}  "
                f"ent={stats['entropy']:.3f}  clip={stats['clip_frac']*100:.1f}%"
            )

        if it % cfg.eval_interval == 0 or it == cfg.iters:
            acc = greedy_accuracy(model, task, task.pairs, cfg)
            print(f"    -> greedy exact on {len(task.pairs)} problems = {acc*100:.1f}%  (elapsed {time.time()-t0:.1f}s)")

    return model, history


@torch.no_grad()
def greedy_accuracy(model: PolicyLM, task: ArithmeticTask, pairs, cfg: Config):
    prompt_ids = torch.tensor([task.encode_prompt(a, b) for a, b in pairs], device=cfg.device)
    comp = model.greedy(prompt_ids, cfg.max_new_tokens)
    a_arr = torch.tensor([a for a, _ in pairs])
    b_arr = torch.tensor([b for _, b in pairs])
    _, exact, _ = task.score(a_arr, b_arr, comp, completion_mask(comp))
    return float(np.mean(exact))


def plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图：pip install matplotlib")
        return

    xs = [h["iter"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].plot(xs, [h["exact"] * 100 for h in history], color="#2a6fdb", linewidth=1.6, label="exact (greedy in-batch)")
    axes[0].plot(xs, [h["well"] * 100 for h in history], color="#8da0cb", linewidth=1.2, linestyle="--", label="well-formed")
    axes[0].set_title("answer accuracy")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, [h["dead"] * 100 for h in history], color="#d95f02", linewidth=1.6)
    axes[1].set_title("zero-advantage groups (all same reward)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("%")
    axes[1].grid(alpha=0.25)

    axes[2].plot(xs, [h["len"] for h in history], color="#1b9e77", linewidth=1.6)
    axes[2].set_title("mean completion length (tokens)")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("tokens")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"learning curve saved to {path}")


# --------------------------------------------------------------------------------------
# 7. 数值断言：把公式和实现对齐
# --------------------------------------------------------------------------------------
def selftest():
    ok = []

    # ① 组内 Z-score：手算 [1,0,0,0]
    adv, dead = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]), 4, eps=0.0)
    std = (0.25 - 0.25 ** 2) ** 0.5          # 总体标准差 = sqrt(0.1875) ≈ 0.433
    assert abs(adv[0].item() - 0.75 / std) < 1e-5, adv
    assert abs(adv[1].item() + 0.25 / std) < 1e-5, adv
    assert abs(adv.sum().item()) < 1e-5, adv          # 组内均值恒为 0
    assert dead == 0.0
    ok.append("group Z-score matches hand calculation (mean=0, std=1)")

    # ② 全同奖励 → 优势全 0，组被判为 dead
    adv2, dead2 = group_advantages(torch.tensor([0.2] * 8), 8)
    assert torch.allclose(adv2, torch.zeros(8)), adv2
    assert dead2 == 1.0
    ok.append("identical rewards give zero advantage (dead group)")

    # ③ KL(k3) 恒非负，k1 可以取负
    torch.manual_seed(0)
    ref = torch.log_softmax(torch.randn(64, 5), -1)
    new = torch.log_softmax(torch.randn(64, 5), -1)
    d = (ref - new)
    k3 = (d.exp() - d - 1.0)
    k1 = -d
    assert k3.min().item() >= -1e-6, k3.min()
    assert k1.min().item() < 0, k1.min()
    ok.append("KL k3 estimator is non-negative; k1 is not")

    # ④ ratio = 1 时，截断目标退化为 A，损失梯度就是 -A
    a = torch.tensor([0.5, -0.5])
    r1 = torch.ones(2)
    assert torch.allclose(clip_objective(r1, a, 0.2, 0.2), a)
    # 把 ratio 设成需要梯度的变量，验证 d(clip_obj)/d(ratio) 在 ratio=1 处等于 A
    ratio = torch.ones(1, requires_grad=True)
    clip_objective(ratio, torch.tensor([1.0]), 0.2, 0.2).backward()
    assert torch.allclose(ratio.grad, torch.tensor([1.0]))
    ratio = torch.ones(1, requires_grad=True)
    clip_objective(ratio, torch.tensor([-1.0]), 0.2, 0.2).backward()
    assert torch.allclose(ratio.grad, torch.tensor([-1.0]))
    ok.append("clip objective gradient = A at ratio=1 (A>0 and A<0)")

    # ⑤ 越界后梯度为 0（A>0 超上限 / A<0 低于下限）
    ratio = torch.tensor([1.5], requires_grad=True)
    clip_objective(ratio, torch.tensor([1.0]), 0.2, 0.2).backward()
    assert ratio.grad.item() == 0.0
    ratio = torch.tensor([0.5], requires_grad=True)
    clip_objective(ratio, torch.tensor([-1.0]), 0.2, 0.2).backward()
    assert ratio.grad.item() == 0.0
    ok.append("clipped region has zero gradient (no reward for overshooting)")

    # ⑥ 规则打分器与 EOS 掩码
    task = ArithmeticTask()
    comp = torch.tensor([[1, 2, EOS], [1, EOS, EOS], [3, 9, 1]])
    mask = completion_mask(comp)
    assert torch.allclose(mask.sum(1), torch.tensor([3.0, 2.0, 3.0]))
    r, exact, well = task.score(torch.tensor([3, 3, 0]), torch.tensor([9, 9, 0]), comp, mask)
    assert exact[0] == 1.0 and well[0] == 1.0
    assert abs(r[1] - (0.2 + 0.8 * 0.5)) < 1e-6, r[1]      # "1" vs "12"：格式分 + 一位匹配
    assert r[2] == 0.0 and well[2] == 0.0                  # 没吐 EOS：格式不合法
    ok.append("rule reward + EOS mask behave as specified")

    # ⑦ 采样后重新算 log 概率，应与采样时记录的一致（同温度）
    torch.manual_seed(0)
    m = PolicyLM()
    prompts = torch.tensor([[3, PLUS, 9, EQ], [4, PLUS, 5, EQ]])
    comp_s, logp_s, mask_s = m.sample(prompts, 4, 3, temperature=1.0)
    logp_r, _ = m.logprobs(prompts.repeat_interleave(4, 0), comp_s, 1.0)
    assert torch.allclose(logp_s * mask_s, logp_r * mask_s, atol=1e-5)
    ok.append("sampling logp == teacher-forcing logp (same temperature)")

    for i, s in enumerate(ok, 1):
        print(f"  [{i}] OK  {s}")
    print(f"selftest passed ({len(ok)} checks)")


# --------------------------------------------------------------------------------------
# 8. 命令行入口
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GRPO from scratch (single file)")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--iters", type=int, default=Config.iters)
    p.add_argument("--batch-prompts", type=int, default=Config.batch_prompts)
    p.add_argument("--group-size", type=int, default=Config.group_size)
    p.add_argument("--temperature", type=float, default=Config.temperature)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--minibatch-size", type=int, default=Config.minibatch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--clip-eps", type=float, default=None, help="同时设置上下截断")
    p.add_argument("--clip-eps-low", type=float, default=Config.clip_eps_low)
    p.add_argument("--clip-eps-high", type=float, default=Config.clip_eps_high)
    p.add_argument("--kl-coef", type=float, default=Config.kl_coef)
    p.add_argument("--entropy-coef", type=float, default=Config.entropy_coef)
    p.add_argument("--no-normalize-std", action="store_true", help="占位：默认就按 Dr. GRPO 之外的原始做法")
    p.add_argument("--drop-dead-groups", action="store_true")
    p.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    p.add_argument("--log-interval", type=int, default=Config.log_interval)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot", default="")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return

    kw = {k: v for k, v in vars(args).items() if k not in ("plot", "selftest", "clip_eps", "no_normalize_std")}
    if args.clip_eps is not None:
        kw["clip_eps_low"] = kw["clip_eps_high"] = args.clip_eps
    kw["normalize_std"] = not args.no_normalize_std
    cfg = Config(**kw)

    model, history = train(cfg)
    if args.plot:
        plot_history(history, args.plot)


if __name__ == "__main__":
    main()
