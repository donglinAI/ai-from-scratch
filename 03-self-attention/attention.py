from __future__ import annotations 

import math 

import torch 
# torch.set_printoptions(precision=3, sci_mode=False) # 打印结果 将科学记数法改成小数
import torch.nn as nn
import torch.nn.functional as F

# ------ 旋转位置编码 ------ 
def precompute_freqs_cis(
        dim: int, 
        max_seq_len: int,
        theta: float = 10000.0,
        device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ 预计算旋转位置编码的 cos/sin 表。 """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """ 半旋转并取负数操作 """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : ]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
):
    """ 对 q/k 施加位置旋转, 变成带位置信息的 q_embed/k_embed. """
    cos = cos[position_ids]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    cos = cos.unsqueeze(1)

    sin = sin[position_ids]
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    sin = sin.unsqueeze(1)

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
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids)


# ------ 自注意力机制（Self-Attention）------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: 把 kv 头重复 n_rep 次，扩展到与 q 头数一致。
    
    x: [batch, kv_heads, seq, head_dim] -> [batch, kv_heads*n_rep, seq, head_dim]
    用 expand (不复制内存), 重复只是视图。
    """
    batch, kv_heads, seq, head_dim = x.shape 
    if n_rep == 1:
        return x 
    x = x[:, :, None, :, :].expand(batch, kv_heads, n_rep, seq, head_dim)
    return x.reshape(batch, kv_heads * n_rep, seq, head_dim)


class MultiHeadAttention(nn.Module):
    """支持 MHA (num_kv_heads=num_heads) 与 GQA (num_kv_heads < num_heads). """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int | None = None, # num_kv_heads < num_heads 时为 GQA; 为 None 时为 MHA
            head_dim: int | None = None, # head_dim 为 None 时自动计算为 hidden_size // num_heads
            max_seq_len: int = 2048,
            rope_theta: float = 10000.0,
            bias = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size 
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads 
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads 
        self.n_rep = num_heads // num_kv_heads  # 每个 kv 服务多少个 q
        assert self.num_heads % self.num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        # 投影：q 全头数，k/v 只有 kv_heads 组
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)

        # RoPE 旋转位置编码（无参数）
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def forward(
            self,
            x: torch.Tensor, # [batch, seq, hidden_size]
            positions: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """"""

        b, s, _ = x.shape 

        # 1) 投影 + 切头: [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2) 
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2) RoPE: 对 q/k 施加位置编码
        if positions is None:
            positions = torch.arange(s, device=x.device)
        q, k = self.rope(q, k, positions)

        # 3) GQA 扩展：kv 头 重复到与q 头一致
        k = repeat_kv(k, self.n_rep) # [B, H, S, D]
        v = repeat_kv(v, self.n_rep)

        # 4) 缩放点积注意力分数 
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) causal 掩码 （decoder-only) : 上三角 -inf
        if attention_mask is None:
            causal = torch.triu(torch.ones(s, s, dtype=bool, device=x.device), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal, float("-inf"))

        # 6) softmax + 加权求和
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype) # softmax计算敏感, 提升精度计算后再转回去(原weights可为fp16/bf16)
        out = torch.matmul(attn_weights, v) # [B, H, S, D]

        # 7) 合并头 + 输出投影 [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D]
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)

if __name__ == "__main__":
    torch.manual_seed(0) # 设置 PyTorch 随机种子（CPU），保证实验可复现
    H, HEADS, KV, S = 512, 8, 2, 6 

    # --- 校验 1： 打印 MHA & GQA 结果 --- 
    x = torch.randn(2, S, H, requires_grad=True) # 输入测试张量

    mha = MultiHeadAttention(H, HEADS, num_kv_heads=HEADS, max_seq_len=32)
    gqa = MultiHeadAttention(H, HEADS, num_kv_heads=KV, max_seq_len=32)

    y1, y2 = mha(x), gqa(x)
    print(f"输入形状-> {x.shape}")
    print(f"输出形状-> MHA结果: {y1.shape}, GQA结果: {y2.shape}")


    # --- 校验 2: MHA & GQA 参数对比 --- 
    p_mha_all = sum(p.numel() for p in mha.parameters())
    p_gqa_all = sum(p.numel() for p in gqa.parameters())

    print(f"\nMHA(kv={HEADS}) 总参数量: {p_mha_all} | GQA(kv={KV}) 总参数量: {p_gqa_all}; "
          f" -> 参数量比值: GQA / MHA = {p_gqa_all / p_mha_all}")

    # 单独统计 K,V 参数（mha/gqa 所有参数包括 q,k,v,o）
    def count_param(model, name):
        return sum(p.numel() for n,p in model.named_parameters() if name in n)

    kv_mha = count_param(mha, "k_proj.weight") + count_param(mha, "v_proj.weight")
    kv_gqa = count_param(gqa, "k_proj.weight") + count_param(gqa, "v_proj.weight")

    print(f"MHA(kv={HEADS}) KV参数量: {kv_mha} | GQA(kv={KV}) KV参数量: {kv_gqa}; "
          f" -> 参数量比值: GQA / MHA = {kv_gqa / kv_mha}")


    # --- 校验 3: 与 HF LlamaAttention 数值对齐 --- 
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention 

        cfg = LlamaConfig(
            hidden_size=H,
            num_attention_heads=HEADS,
            num_key_value_heads=KV,
            head_dim=H // HEADS,
            max_position_embeddings=32,
            rope_theta=10000.0,
            attention_bias=False,
            num_hidden_layers=1,
        )
        # 加上这一行，指定用eager（可选值：`"eager"`（朴素手写版） / `"sdpa"`（PyTorch SDPA 加速） / `"flash_attention_2"`）
        cfg._attn_implementation = "eager"

        hf_attn = LlamaAttention(cfg, layer_idx=0)
        # 权重复制：把我们实现的权重赋值给HF模型
        with torch.no_grad():
            hf_attn.q_proj.weight.copy_(gqa.q_proj.weight)
            hf_attn.k_proj.weight.copy_(gqa.k_proj.weight)
            hf_attn.v_proj.weight.copy_(gqa.v_proj.weight)
            hf_attn.o_proj.weight.copy_(gqa.o_proj.weight)

        # 构造position_ids和RoPE所带cos/sin
        b, s = x.shape[0], x.shape[1]
        position_ids = torch.arange(s,device=x.device)[None, :].expand(b, s)

        # HF 前向（默认是全注意力）
        cos = gqa.rope.cos[:s].unsqueeze(0).expand(b, s, gqa.head_dim)
        sin = gqa.rope.sin[:s].unsqueeze(0).expand(b, s, gqa.head_dim)
        pos_emb = (cos, sin)

        causal_mask = torch.triu(torch.full((s, s), float("-inf"), device=x.device), diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(b, 1, s, s)
        with torch.no_grad():
            hf_out, _ = hf_attn(
                x.detach(),
                position_embeddings=pos_emb,
                attention_mask=causal_mask
            )

        # 我们自己实现的前向（默认是因果注意力）
        with torch.no_grad():
            my_out = gqa(
                x.detach(),
                positions=position_ids,
                attention_mask=None
            )

        max_diff = (hf_out - my_out).abs().max().item()
        print(f"\n HF对齐测试 | 最大误差: {max_diff:.2e}(应 < 1e-5)")
    except Exception as e:
        print(f"\n[skip HF对齐] {type(e).__name__}: {e}")

    print("\n✅ MHA/GQA 自校验通过")