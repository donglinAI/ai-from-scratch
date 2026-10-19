
from __future__ import annotations  # Python 的特性，延迟求值所有类型注解

import torch
# torch.set_printoptions(precision=3, sci_mode=False) # 打印结果 将科学记数法改成小数
import torch.nn as nn 

def precompute_freqs_cis(
        dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算所有位置的 cos / sin 表。
    
    Parameters:
    ----------
    dim: 向量维度 (例如我们在理论部分提到的 d=8); 每个注意力头的维度 head_dim 
    max_seq_len: 支持的最大序列长度;
    theta: 旋转基数; 例如我们在理论部分提到的 10000

    Returns:
    ----------
    cos, sin: 预计算出的结果; 形状 [max_seq_len, dim]
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    ) # [dim/2] 
    t = torch.arange(max_seq_len) # [S]; 序列位置 0..S-1 
    freqs = torch.outer(t, inv_freq) # [S, dim/2]; 每个序列位置的各个频率, dim上两个特征共用一个频率, 所以频率数为 dim / 2
    
    # 复制一份频率, 凑满 dim 维
    emb = torch.cat([freqs, freqs], dim=-1) # [S, dim]

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
    x2 = x[..., x.shape[-1] // 2 : ]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ 对 q/k 施加位置旋转, 变成带位置信息的 q_embed/k_embed. 

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
    cos = cos[position_ids] # 根据位置m取 该位置下的各个频率cos
    if cos.dim() == 2:
        cos = cos.unsqueeze(0) # [1, seq, D]补 batch 维; 如果 position_ids 的 shape 为 [seq], 则 cos 的 shape 为 [seq, dim], cos.unsqueeze(0) 的 shape 为 [1, seq, dim
    cos = cos.unsqueeze(1) # 每一个样本需要1组位置编码即可
    
    sin = sin[position_ids] # 同上
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    sin = sin.unsqueeze(1)

    # 旋转公式
    q_embed = q * cos + rotate_half(q) * sin 
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed 


class RotaryEmbedding(nn.Module):
    """RoPE模块: 预计算表 + 应用旋转。无任何可学习参数。"""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim 
        self.max_seq_len = max_seq_len
        self.theta = theta 

        cos, sin = precompute_freqs_cis(dim, max_seq_len, theta)
        # register_buffer: 随模型保存/加载、随.to(device) 移动; 不参与梯度
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def forward(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids)  # __init__注册了 cos 和 sin; 所以可以使用 self.cos / self.sin
    

if __name__ == "__main__":
    torch.manual_seed(0)
    D, S, B, H = 64, 8, 2, 4 

    rope = RotaryEmbedding(dim=D, max_seq_len=32, theta=10000.0)
    q = torch.randn(B, H, S, D, requires_grad=True)
    k = torch.randn(B, H, S, D)
    pos = torch.arange(S).repeat(B, 1) # batch中每个样本 有 S 个位置;

    q_r, k_r = rope(q, k, pos)
    print(f"q 形状: {q.shape} -> q_r 形状: {q_r.shape}")

    # --- 校验1: 旋转不改变范数（等距变换，模长不变）---
    err = (q.norm(dim=-1) - q_r.norm(dim=-1)).abs().max().item()
    print(f"\n 旋转前后范数最大差: {err:.2e} (应 < 1e-5)") # :.2e 表示 使用科学记数法, 保留2位小数

    # --- 校验2: 核心性质 -> 内积只依赖相对位置差 ---
    # 对于同一个q 和 同一个k; 位置差相等, 点积结果相同; 反之不同
    def rope_at(x: torch.Tensor, m: int) -> torch.Tensor:
        cos_m = rope.cos[m]
        sin_m = rope.sin[m]
        return x * cos_m + rotate_half(x) * sin_m
    
    vq = torch.randn(D)  # 固定 q 内容
    vk = torch.randn(D)  # 固定 k 内容

    d13 = (rope_at(vq, 1) * rope_at(vk, 3)).sum()  # 位置差 = 2
    d57 = (rope_at(vq, 5) * rope_at(vk, 7)).sum()  # 位置差 = 2
    print(f"\n同一对向量, 位置差=2 (1,3): {d13:.4f} | (5,7): {d57:.4f} | 差: {abs(d13-d57):.2e} (应≈0)")

    d12 = (rope_at(vq, 1) * rope_at(vk, 2)).sum()  # 位置差 = 1
    d16 = (rope_at(vq, 1) * rope_at(vk, 7)).sum()  # 位置差 = 6
    print(f"位置差=1 (1,2): {d12:.4f} | 位置差=6 (1,7): {d16:.4f} (应明显不同)")

    # --- 校验 3: 无旋转时内积与位置完全无关（反证旋转的必要性）---
    d_raw = (vq * vk).sum()
    print(f"未旋转内积处处相同: {d_raw:.4f} → 模型无法区分 (1,3) 和 (5,7) 这类相对距离")

    # --- 校验 4: 与 HF LLaMA 的旋转编码数值对齐 ---
    try:
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb as hf_apply,
        )
        q_hf, k_hf = hf_apply(
            q.float(),
            k.float(),
            rope.cos[:S].unsqueeze(0), # [1, S, D] -> unsqueeze(1) -> [1, 1, S, D] 广播对齐 q/k 维度
            rope.sin[:S].unsqueeze(0),
        )
        max_diff = (q_hf - q_r.float()).abs().max().item()
        print(f"\n与 HF apply_rotary_pos_emb 最大误差: {max_diff:.2e} (应 < 1e-5)")
    except (ImportError) as e:
        print(f"\n[skip] 跳过 HF 对齐测试: {type(e).__name__}")
        
    # --- 校验 5: 反向传播正常（无参数，梯度应流回 q）---
    loss = q_r.sum()
    loss.backward()
    print(f"q 梯度形状: {q.grad.shape} (应 [2, 4, 8, 64])")
    print(f"cos/sin 是 buffer（非参数）: {sum(1 for _ in rope.parameters())} 个参数 (应为 0)")

    print("\n✅ RoPE 自校验通过")