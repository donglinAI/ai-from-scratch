from __future__ import annotations  # Python 的特性，延迟求值所有类型注解
import math
import torch
import torch.nn as nn


def llama3_inv_freq(
    dim: int,
    theta: float = 500000.0,
    factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    original_max_seq_len: int = 8192,
    device=None,
) -> torch.Tensor:
    """Llama-3.1 三段式缩放后的角频率 f_i'。

    Parameters:
    ----------
    dim: 每个注意力头的维度 head_dim (例如 128); 频率槽位数为 dim / 2
    theta: 旋转基数; Llama-3.1 为 500000
    factor: 低频段的压缩倍数 alpha; Llama-3.1 为 8
    low_freq_factor / high_freq_factor: 决定两条分界波长的系数 ell / h
    original_max_seq_len: 原始训练长度 L; 对应 HF config 的 original_max_position_embeddings

    Returns:
    ----------
    inv_freq: 缩放后的角频率; 形状 [dim/2], 单位 rad / token

    三段规则 (lambda_i = 2*pi / f_i 为波长):
        lambda_i > L / ell  -> 低频段, f_i / alpha
        lambda_i < L / h    -> 高频段, f_i 不动
        其余                -> 中频段, (1-s) * f_i/alpha + s * f_i
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )  # [dim/2]; 未缩放的 f_i = theta^(-2i/dim)
    wavelen = 2 * math.pi / inv_freq  # [dim/2]; 波长 lambda_i, 单位 token / 圈

    wavelen_low = original_max_seq_len / low_freq_factor    # 8192; 波长比它长 -> 低频段
    wavelen_high = original_max_seq_len / high_freq_factor  # 2048; 波长比它短 -> 高频段

    is_low = wavelen > wavelen_low    # 低频段掩码
    is_high = wavelen < wavelen_high  # 高频段掩码
    is_mid = ~(is_low | is_high)      # 中频段掩码 (过渡带)

    # 插值权重 s: L/lambda 表示"该维度在原训练长度内转的圈数",
    # 把圈数从 [ell, h] 归一化到 [0, 1]; lambda 越大 s 越小, 压得越狠
    s = (original_max_seq_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    s = s.clamp(0.0, 1.0)  # 只在中频段被取用, clamp 属于防御性写法

    # 两端分支: 低频段除以 alpha, 高频段原样保留
    inv_freq_scaled = torch.where(is_low, inv_freq / factor, inv_freq)
    # 中频段: 在 f/alpha 与 f 之间做凸组合, 两端点与上下两段自动衔接
    inv_freq_mid = (1.0 - s) * inv_freq / factor + s * inv_freq
    return torch.where(is_mid, inv_freq_mid, inv_freq_scaled)  # [dim/2]


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    rope_scaling: dict | None = None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算所有位置的 cos / sin 表。

    Parameters:
    ----------
    dim: 向量维度; 每个注意力头的维度 head_dim
    max_seq_len: 支持的最大序列长度
    theta: 旋转基数; 标准 RoPE 常用 10000, Llama-3.1 为 500000
    rope_scaling:  None 表示标准 RoPE; 传入 {"rope_type": "llama3", ...} 则启用三段式缩放

    Returns:
    ----------
    cos, sin: 预计算出的结果; 形状 [max_seq_len, dim]
    """
    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        inv_freq = llama3_inv_freq(
            dim,
            theta=theta,
            factor=rope_scaling.get("factor", 8.0),
            low_freq_factor=rope_scaling.get("low_freq_factor", 1.0),
            high_freq_factor=rope_scaling.get("high_freq_factor", 4.0),
            original_max_seq_len=rope_scaling.get("original_max_seq_len", 8192),
            device=device,
        )  # [dim/2]; 三段式缩放后的频率
    else:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )  # [dim/2]; 标准 RoPE, 不做任何缩放

    t = torch.arange(max_seq_len)  # [S]; 序列位置 0..S-1
    freqs = torch.outer(t, inv_freq)  # [S, dim/2]; 每个位置的各个频率, 两个特征共用一个频率

    # 复制一份频率, 凑满 dim 维
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """半旋转并取负数操作

    Example:
        >>> import torch
        >>> x = torch.tensor([[1,2,3,4]])
        >>> rotate_half(x)
        torch.tensor([[-3.,-4.,1.,2.]])
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 q/k 施加位置旋转, 变成带位置信息的 q_embed/k_embed。

    Parameters:
    ----------
    q, k: [batch, heads, seq, head_dim]; 为了后续可将该函数植入真实模型, 所以直接支持多头+batch形式
    cos, sin: [max_seq_len, head_dim]
    position_ids: [batch, seq] 或 [seq]

    Returns:
    ----------
    q_embed/k_embed: 带位置信息的向量
    """
    # 按 position_ids 查表 -> [S, D] (1D 位置) 或 [B, S, D] (2D 位置)
    cos = cos[position_ids]  # 根据位置 m 取该位置下的各个频率 cos
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)  # [1, seq, D]; 补 batch 维
    cos = cos.unsqueeze(1)  # 每一个样本需要 1 组位置编码即可

    sin = sin[position_ids]  # 同上
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    sin = sin.unsqueeze(1)

    # 旋转公式
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE 模块: 预计算表 + 应用旋转。无任何可学习参数。"""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = precompute_freqs_cis(dim, max_seq_len, theta, rope_scaling)
        # register_buffer: 随模型保存/加载、随.to(device) 移动; 不参与梯度
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids)


if __name__ == "__main__":
    dim = 128  # Llama-3.1 的 head_dim, 共 64 个频率槽位
    scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192,
    }

    # 1) 看三段各分到多少个槽位
    inv_freq = llama3_inv_freq(dim)  # 默认参数即 Llama-3.1 配置
    f_raw = 1.0 / (500000.0 ** (torch.arange(0, dim, 2).float() / dim))
    lam = 2 * math.pi / f_raw
    n_keep = int((lam < 2048).sum())
    n_div = int((lam > 8192).sum())
    print(f"三段槽位数: 不动 {n_keep} / 插值 {dim // 2 - n_keep - n_div} / 除以8 {n_div}")

    # 2) 打印几个代表维度, 看 f -> f' 的变化 (k 为波长拉伸倍数)
    for i in (28, 29, 32, 35, 63):
        print(f"i={i:2d}  lambda={lam[i]:12.3f}  f={f_raw[i]:.6e}  "
              f"f'={inv_freq[i]:.6e}  k={float(f_raw[i] / inv_freq[i]):.4f}")

    # 3) 走一遍完整的前向: 预计算 cos/sin, 再对 q/k 施加旋转
    rope = RotaryEmbedding(dim, max_seq_len=16, theta=500000.0, rope_scaling=scaling)
    q = torch.randn(1, 2, 4, dim)  # [batch=1, heads=2, seq=4, head_dim=128]
    k = torch.randn(1, 2, 4, dim)
    q_embed, k_embed = rope(q, k, torch.arange(4))
    print("q_embed:", q_embed.shape, " k_embed:", k_embed.shape)
    print("旋转前后模长一致:", torch.allclose(q.norm(dim=-1), q_embed.norm(dim=-1), atol=1e-5))
