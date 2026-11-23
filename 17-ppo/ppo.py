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

import argparse
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # gymnasium 为推荐依赖，gym 作为旧版兼容
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore


# --------------------------------------------------------------------------------------
# 超参数
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_updates: int = 150          # 迭代轮数（每轮采一批数据，更新完即丢弃）
    rollout_steps: int = 2048         # 每轮采样步数
    gamma: float = 0.99               # 折扣因子
    lam: float = 0.95                 # GAE 的 lambda
    clip_eps: float = 0.2             # PPO 截断范围 [1-eps, 1+eps]
    lr: float = 3e-4
    epochs: int = 10                  # 同一批数据重复训练轮数
    minibatch_size: int = 64
    value_coef: float = 0.5           # 价值损失权重
    entropy_coef: float = 0.01        # 熵奖励权重（鼓励探索）
    max_grad_norm: float = 0.5
    target_kl: float = 0.015          # 近似 KL 超过该值提前结束本批更新
    hidden_dim: int = 64
    eval_interval: int = 10
    eval_episodes: int = 5
    log_interval: int = 1
    device: str = "cpu"


# --------------------------------------------------------------------------------------
# 1. 网络：Actor-Critic
# --------------------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """共享特征主干，两个输出头：动作分布 + 状态价值。

    - 策略头输出 logits，经 Categorical 得到动作分布 π(a|s)
    - 价值头输出标量 V(s)，只有 1 维
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        # 正交初始化：策略头小增益输出接近均匀分布，价值头增益 1.0
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """返回 (动作分布, 状态价值)。"""
        h = self.trunk(obs)
        logits = self.policy_head(h)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_head(h).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """采样阶段：按概率分布采样动作（不是 argmax），保留探索性。"""
        dist, value = self.forward(obs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


# --------------------------------------------------------------------------------------
# 2. 采样缓存
# --------------------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.logprobs = [], [], []
        self.rewards, self.values, self.values_next, self.terminals = [], [], [], []

    def add(self, obs, action, logp, reward, value, value_next, terminal):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logp)
        self.rewards.append(reward)
        self.values.append(value)
        self.values_next.append(value_next)
        self.terminals.append(terminal)

    def stack(self, device):
        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)

        obs = t(self.obs)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.long, device=device)
        logprobs = t(self.logprobs)
        rewards = t(self.rewards)
        values = t(self.values)
        values_next = t(self.values_next)
        terminals = t(self.terminals)
        return obs, actions, logprobs, rewards, values, values_next, terminals


# --------------------------------------------------------------------------------------
# 3. GAE：广义优势估计
# --------------------------------------------------------------------------------------
def compute_gae(rewards, values, values_next, terminals, gamma, lam):
    r"""逐时刻倒序递推：

    δ_t = r_t + γ V(s_{t+1})(1-done_t) - V(s_t)
    A_t = δ_t + γλ(1-done_t) A_{t+1}
    """
    n = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(n)):
        mask = 1.0 - terminals[t]
        delta = rewards[t] + gamma * values_next[t] * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv

    returns = advantages + values  # 价值网络回归目标
    return advantages, returns


# --------------------------------------------------------------------------------------
# 4. PPO 更新
# --------------------------------------------------------------------------------------
def ppo_update(model, optimizer, buffer, cfg):
    obs, actions, old_logprobs, rewards, values, values_next, terminals = buffer.stack(cfg.device)
    advantages, returns = compute_gae(rewards, values, values_next, terminals, cfg.gamma, cfg.lam)

    # 优势标准化：与 GRPO 的组内归一化同理，都是把"信号"变成可比的分差
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.shape[0]
    idx = torch.arange(batch_size, device=cfg.device)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "n": 0}

    for _ in range(cfg.epochs):
        idx = idx[torch.randperm(batch_size, device=cfg.device)]
        for start in range(0, batch_size, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]

            dist, new_values = model(obs[mb])
            new_logprobs = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            log_ratio = new_logprobs - old_logprobs[mb]
            ratio = log_ratio.exp()

            adv = advantages[mb]

            # 未裁剪目标 与 裁剪目标，取较小者 → 悲观下界
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, returns[mb])

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            # 近似 KL（无偏、恒正）：(r-1) - log r
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
            clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n"] += 1

        if stats["n"] and stats["kl"] / stats["n"] > cfg.target_kl:
            break  # 新旧策略偏离过大，提前停止本批更新

    for k in ("policy_loss", "value_loss", "entropy", "kl", "clip_frac"):
        stats[k] = stats[k] / max(stats["n"], 1)
    return stats, advantages


# --------------------------------------------------------------------------------------
# 5. 环境交互（兼容 gymnasium / gym 两套 reset-step 接口）
# --------------------------------------------------------------------------------------
def env_reset(env, seed=None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, info


def obs_to_tensor(obs, device):
    return torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)


@torch.no_grad()
def evaluate(model, env, cfg):
    model.eval()
    rewards = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env_reset(env)
        done, ep_ret = False, 0.0
        while not done:
            dist, _ = model(obs_to_tensor(obs, cfg.device))
            action = int(dist.probs.argmax().item())  # 评测用贪心
            obs, reward, terminated, truncated, _ = env_step(env, action)
            done = terminated or truncated
            ep_ret += reward
        rewards.append(ep_ret)
    model.train()
    return float(np.mean(rewards))


@torch.no_grad()
def rollout(model, env, obs, cfg, ep_returns):
    """采样 rollout_steps 步；返回 buffer 与新起点 obs。

    ep_returns 是长度至少为 1 的列表：最后一个元素是"正在进行中"这一局
    的累计回报，每结束一局就 append 一个新的 0.0。
    """
    buf = RolloutBuffer()
    for _ in range(cfg.rollout_steps):
        o_t = obs_to_tensor(obs, cfg.device)
        action, logp, value = model.act(o_t)

        a = int(action.item())
        next_obs, reward, terminated, truncated, _ = env_step(env, a)

        # 只有真终止才是"价值为 0"；truncated（时间上限）仍要自举
        terminal = float(terminated)
        v_next = 0.0
        if not terminated:
            _, v_next_t = model(obs_to_tensor(next_obs, cfg.device))
            v_next = float(v_next_t.item())

        buf.add(obs, a, float(logp.item()), reward, float(value.item()), v_next, terminal)

        ep_returns[-1] += reward
        if terminated or truncated:
            ep_returns.append(0.0)
            obs, _ = env_reset(env)
        else:
            obs = next_obs

    return buf, obs


# --------------------------------------------------------------------------------------
# 6. 训练主循环
# --------------------------------------------------------------------------------------
def train(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    obs, _ = env_reset(env, seed=cfg.seed)
    best_eval = -float("inf")
    history = []
    ep_returns = [0.0]           # 当前 episode 累计回报（末尾为进行中的 episode）
    finished_returns = []
    start = time.time()

    for update in range(1, cfg.total_updates + 1):
        n_before = len(finished_returns)

        buf, obs = rollout(model, env, obs, cfg, ep_returns)

        # 结算本轮完成的 episode
        finished_returns.extend(ep_returns[:-1])
        new_eps = finished_returns[n_before:]
        ep_returns = [ep_returns[-1]]

        stats, _ = ppo_update(model, optimizer, buf, cfg)

        mean_ret = float(np.mean(new_eps)) if new_eps else float("nan")
        history.append((update, mean_ret, stats))

        if update % cfg.log_interval == 0:
            print(
                f"[upd {update:4d}] ep_ret={mean_ret:7.1f}  "
                f"pi_loss={stats['policy_loss']:+.3f}  v_loss={stats['value_loss']:.3f}  "
                f"ent={stats['entropy']:.3f}  kl={stats['kl']:.4f}  clip={stats['clip_frac']*100:.1f}%"
            )

        if update % cfg.eval_interval == 0 or update == cfg.total_updates:
            score = evaluate(model, eval_env, cfg)
            print(f"    -> eval greedy reward = {score:.1f}  (elapsed {time.time()-start:.1f}s)")
            if score > best_eval:
                best_eval = score
                torch.save({"model": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}, "ppo_best.pt")
            threshold = getattr(env.spec, "reward_threshold", None) or 475.0
            if score >= threshold:
                print(f"Solved in {update} updates (threshold={threshold}).")
                break

    env.close()
    eval_env.close()
    return model, history, best_eval


def plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图：pip install matplotlib")
        return

    xs = [h[0] for h in history if not np.isnan(h[1])]
    ys = [h[1] for h in history if not np.isnan(h[1])]
    kl = [h[2]["kl"] for h in history if not np.isnan(h[1])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(xs, ys, color="#2a6fdb", linewidth=1.6)
    axes[0].set_title("episode return per update")
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("return")
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, kl, color="#d95f02", linewidth=1.6)
    axes[1].set_title("approx KL of policy update")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("KL")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"learning curve saved to {path}")


def parse_args():
    p = argparse.ArgumentParser(description="PPO from scratch (single file)")
    p.add_argument("--env-id", default=Config.env_id)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--updates", type=int, default=Config.total_updates, dest="total_updates")
    p.add_argument("--rollout-steps", type=int, default=Config.rollout_steps)
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
    p.add_argument("--plot", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if not k == "plot"})
    model, history, best = train(cfg)
    print(f"training finished. best greedy eval return = {best:.1f}")
    if args.plot:
        plot_history(history, args.plot)


if __name__ == "__main__":
    main()
