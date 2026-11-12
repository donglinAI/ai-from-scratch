#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Llama 3.1 8B 从零复现（单文件 · 推理版）

一个文件跑完「ModelScope 下载权重 → 预填充 → KV Cache 增量解码 → 采样输出」。

文件组织（从上往下读就是推理链路）：
    1. 基础件        RoPE / RMSNorm / SwiGLU / repeat_kv
    2. 配置          Llama 3.1 8B 超参 + 读官方 config.json
    3. RoPE 缩放     llama3 型三段式频率缩放（128k 上下文的唯一秘密）
    4. KV Cache      预分配静态表，写入切片、读取视图
    5. 模型          Attention(GQA+RoPE+mask+cache) / DecoderLayer / ForCausalLM
    6. 权重加载      ModelScope safetensors 分片 -> 本模型
    7. 采样与生成    greedy / 温度 / top-k / top-p，全程增量解码
    8. 自检          与 transformers 的 LlamaForCausalLM 逐位对齐
    9. 运行          python 脚本直接跑一次推理

用法：
    # 1) 自检：不下载权重，用 tiny 模型和 HF 逐位对齐（CPU 十几秒）
    python llama31_inference.py --check

    # 2) 真实推理（默认从 ModelScope 拉 Meta-Llama-3.1-8B-Instruct）
    python llama31_inference.py --prompt "用一句话解释 RoPE"

    # 3) 指定本地权重目录 / 其他 ModelScope 模型 id
    python llama31_inference.py --prompt "用一句话解释 RoPE" --model /root/autodl-tmp/cache/modelscope/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master/

依赖：torch / transformers / safetensors / modelscope。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 默认模型：ModelScope 上的 Llama-3.1-8B-Instruct
DEFAULT_MODEL_ID = "LLM-Research/Meta-Llama-3.1-8B-Instruct"


# =====================================================================
# 1. 基础件
# =====================================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """半旋转取负：[x1, x2] -> [-x2, x1]。

    Example
    -------
    >>> rotate_half(torch.tensor([[1., 2., 3., 4.]]))
    tensor([[-3., -4.,  1.,  2.]])
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    """均方根归一化：只有缩放 weight，没有平移 bias（与 HF 命名一致）。

    计算抬到 fp32：bf16 下直接在低精度里求平方均值 + 开方会累积误差，长序列上肉眼可见。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() / rms * x32).to(x.dtype)


class SwiGLU(nn.Module):
    """门控前馈层：down(silu(gate(x)) * up(x))。"""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA：把 kv 头重复 n_rep 次扩到 q 头数（用 expand，只建视图）。

    [B, kv_heads, S, D] -> [B, kv_heads*n_rep, S, D]
    """
    b, kv_heads, s, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, s, d)
    return x.reshape(b, kv_heads * n_rep, s, d)


# =====================================================================
# 2. 配置（Llama 3.1 8B）
# =====================================================================
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


@dataclass
class Llama31Config:
    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int | None = None
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: dict | None = field(default_factory=lambda: dict(ROPE_SCALING_LLAMA3))
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None

    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0

    @property
    def n_rep(self) -> int:
        """GQA：每个 kv 头服务几个 q 头。"""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json_file(cls, path: str | Path) -> "Llama31Config":
        """读官方 config.json（transformers v5 格式：rope 参数统一在 rope_parameters 里）。"""
        d = json.loads(Path(path).read_text())
        rope = dict(d.get("rope_parameters") or {})
        theta = float(rope.pop("rope_theta", d.get("rope_theta", 10000.0)))
        if rope.get("rope_type", "default") == "default":
            rope = None
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim"),
            max_position_embeddings=d["max_position_embeddings"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=theta,
            rope_scaling=rope,
            attention_bias=d.get("attention_bias", False),
            mlp_bias=d.get("mlp_bias", False),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            pad_token_id=d.get("pad_token_id"),
        )

    def to_hf_config(self, **overrides):
        """反向生成 HF 的 LlamaConfig（仅自检对齐用）。"""
        from transformers import LlamaConfig

        rope = dict(self.rope_scaling or {})
        rope["rope_theta"] = self.rope_theta

        kwargs = dict(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size, num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim, max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps, attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias, tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id, rope_parameters=rope,
        )
        kwargs.update(overrides)
        return LlamaConfig(**kwargs)


def tiny_config(**kw) -> Llama31Config:
    """自检用的迷你配置（结构一样，只是维度小，CPU 秒级）。"""
    base = dict(vocab_size=1024, hidden_size=256, intermediate_size=688, num_hidden_layers=4,
                num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                max_position_embeddings=2048)
    base.update(kw)
    return Llama31Config(**base)


# =====================================================================
# 3. RoPE：llama3 型频率缩放
# =====================================================================
def compute_inv_freq_llama3(dim: int, theta: float = 500000.0, factor: float = 8.0,
                            low_freq_factor: float = 1.0, high_freq_factor: float = 4.0,
                            original_max_position_embeddings: int = 8192,
                            device=None) -> torch.Tensor:
    """按波长分三段处理逆频率，与 transformers `_compute_llama3_parameters` 等价。

    波长 wavelen = 2π / inv_freq（某维度转一圈要走多少 token）：
        wavelen >  8192/1 = 8192 : 低频，周期太长 -> inv_freq / 8（把周期拉长）
        wavelen <  8192/4 = 2048 : 高频，局部区分能力靠它 -> 不动
        中间                     : 线性插值，避免两段交界处断层
    """
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (idx / dim))

    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1.0 - smooth) * inv_freq_scaled / factor + smooth * inv_freq_scaled
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_scaled)


class RotaryEmbedding(nn.Module):
    """RoPE：预计算 cos/sin 表 + 按位置施加旋转。无任何可学习参数。"""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0,
                 rope_scaling: dict | None = None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope_scaling = dict(rope_scaling or {})

        if self.rope_scaling:
            inv_freq = compute_inv_freq_llama3(
                dim, theta=theta,
                factor=self.rope_scaling.get("factor", 8.0),
                low_freq_factor=self.rope_scaling.get("low_freq_factor", 1.0),
                high_freq_factor=self.rope_scaling.get("high_freq_factor", 4.0),
                original_max_position_embeddings=self.rope_scaling.get(
                    "original_max_position_embeddings", 8192),
            )
        else:
            idx = torch.arange(0, dim, 2, dtype=torch.float32)
            inv_freq = 1.0 / (theta ** (idx / dim))

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                  # [S, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)           # [S, dim]：半维复制
        # 表固定 fp32：只随 device 迁移，不参与梯度、不写进权重文件
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """q, k : [B, heads, S, head_dim]；position_ids : [B, S]（绝对位置）。"""
        cos = self.cos[position_ids].unsqueeze(1).to(q.dtype)   # [B, 1, S, head_dim]
        sin = self.sin[position_ids].unsqueeze(1).to(q.dtype)
        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin
        return q_embed, k_embed


# =====================================================================
# 4. KV Cache
# =====================================================================
class KVCache:
    """预分配静态缓存 [B, kv_heads, max_seq_len, head_dim]，写入用切片、读取用视图。

    8B / 128k / bf16 显存：2 × 32 层 × 8 头 × 131072 × 128 × 2B ≈ 17.2 GB（16 GiB）。
    """

    def __init__(self, num_layers: int, batch_size: int, num_kv_heads: int,
                 max_seq_len: int, head_dim: int, dtype=torch.float32, device=None):
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """写入本步增量，返回"历史 + 当前"的 K/V 视图（不拷贝）。"""
        end = self.seq_len + k.shape[2]
        if end > self.max_seq_len:
            raise IndexError(f"KV Cache 容量不足：需要 {end} > {self.max_seq_len}")
        self.k[layer_idx][:, :, self.seq_len:end] = k
        self.v[layer_idx][:, :, self.seq_len:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, n: int = 1):
        self.seq_len += n

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def __repr__(self):
        return (f"KVCache(filled={self.seq_len}/{self.max_seq_len}, "
                f"mem={self.memory_bytes() / 2**20:.1f} MiB)")


# =====================================================================
# 5. 模型
# =====================================================================
class Llama31Attention(nn.Module):
    """GQA + RoPE + 因果 mask + KV Cache。"""

    def __init__(self, cfg: Llama31Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(self, x: torch.Tensor, rotary: RotaryEmbedding,
                position_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        b, s, _ = x.shape

        # 1) 投影 + 切头：[B, S, H*D] -> [B, H, S, D]
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2) RoPE：只旋转 q/k，v 不动
        q, k = rotary(q, k, position_ids)

        # 3) KV Cache：写入本步增量，取回历史 + 当前
        past = cache.seq_len if cache is not None else 0
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        kv_len = past + s

        # 4) GQA：kv 头视角 -> q 头视角
        k = repeat_kv(k, self.n_rep)                    # [B, H, kv_len, D]
        v = repeat_kv(v, self.n_rep)

        # 5) 缩放点积打分
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling   # [B, H, S, kv_len]

        # 6) 因果 mask：query 绝对位置 = past + i，只能看 j <= past + i
        q_pos = torch.arange(past, past + s, device=x.device)
        k_pos = torch.arange(kv_len, device=x.device)
        allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        mask = torch.zeros(s, kv_len, dtype=scores.dtype, device=x.device)
        mask.masked_fill_(~allowed, torch.finfo(scores.dtype).min)
        scores = scores + mask.view(1, 1, s, kv_len)

        # 7) softmax 在 fp32 上算再转回，数值更稳
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)                                    # [B, H, S, D]

        # 8) 合并头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Llama31DecoderLayer(nn.Module):
    """pre-norm 残差：x = x + Attn(Norm(x)); x = x + MLP(Norm(x))。"""

    def __init__(self, cfg: Llama31Config, layer_idx: int):
        super().__init__()
        self.self_attn = Llama31Attention(cfg, layer_idx)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, rotary, position_ids, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), rotary, position_ids, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama31Model(nn.Module):
    def __init__(self, cfg: Llama31Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([Llama31DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
            rope_scaling=cfg.rope_scaling,
        )

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None,
                cache: KVCache | None = None) -> torch.Tensor:
        h = self.embed_tokens(input_ids)                # [B, S, hidden]
        b, s, _ = h.shape
        past = cache.seq_len if cache is not None else 0

        if position_ids is None:
            # decode 时位置必须从 past 接着数，否则 RoPE 相位错位
            position_ids = torch.arange(past, past + s, device=h.device).unsqueeze(0).expand(b, -1)

        for layer in self.layers:
            h = layer(h, self.rotary_emb, position_ids, cache)

        if cache is not None:
            cache.advance(s)
        return self.norm(h)


class Llama31ForCausalLM(nn.Module):
    def __init__(self, cfg: Llama31Config):
        super().__init__()
        self.cfg = cfg
        self.model = Llama31Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, position_ids=None, cache=None) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids, cache)
        return self.lm_head(hidden)                     # [B, S, vocab]

    # ---------------- 便捷构造 / 加载 ----------------
    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device="cpu", dtype=torch.float32) -> "Llama31ForCausalLM":
        """从 ckpt 目录（config.json + safetensors 分片）构建并加载权重。

        直接按目标 dtype 建模型：8B 权重同时放 fp32 和 bf16 会多吃一倍内存。
        """
        model_dir = Path(model_dir)
        cfg = Llama31Config.from_json_file(model_dir / "config.json")

        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = cls(cfg)
        finally:
            torch.set_default_dtype(prev)

        model = model.to(device=device)
        state = load_checkpoint(model_dir, dtype=dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"权重缺失，例如 {missing[:5]}")
        if unexpected:
            print(f"[warn] 有 {len(unexpected)} 个未使用 key，例如 {unexpected[:5]}")
        model.eval()
        return model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def new_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg.num_hidden_layers, batch_size, self.cfg.num_key_value_heads,
                       max_seq_len, self.cfg.head_dim, p.dtype, p.device)


# =====================================================================
# 6. 权重加载（ModelScope safetensors 分片 -> 本模型）
# =====================================================================
def load_checkpoint(ckpt: str | Path, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """读 ckpt 目录下的分片权重：model.safetensors.index.json + model-0000x-of-0000y.safetensors。"""
    from safetensors.torch import load_file

    ckpt = Path(ckpt)
    index = ckpt / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    sd = {}
    for shard in sorted(set(weight_map.values())):
        for k, v in load_file(str(ckpt / shard), device="cpu").items():
            sd[k] = v.to(dtype) if dtype is not None else v   # CPU 上直接落成目标精度
    return sd


# =====================================================================
# 7. 采样与生成
# =====================================================================
def sample_next_token(logits: torch.Tensor, temperature: float = 0.0,
                      top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """logits [B, vocab] -> [B]。temperature=0 即 greedy（确定性）。"""
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        drop = cum - probs > top_p                     # 累计概率已超 top_p 的尾部全丢
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(model: Llama31ForCausalLM, input_ids: torch.Tensor, max_new_tokens: int = 64,
             temperature: float = 0.0, top_k: int | None = None, top_p: float | None = None,
             eos_token_ids: int | list[int] | None = None, use_cache: bool = True,
             seed: int | None = None, verbose: bool = False):
    """返回 [B, prompt_len + 新生成长度] 的 token ids；verbose 打印 prefill/decode 吞吐。"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_ids = torch.tensor(eos_token_ids, device=input_ids.device) if eos_token_ids else None

    device = next(model.parameters()).device
    ids = input_ids.to(device)
    b, prompt_len = ids.shape

    cache = model.new_cache(b, prompt_len + max_new_tokens) if use_cache else None

    t0 = time.time()
    logits = model(ids, cache=cache)[:, -1, :]          # prefill：一次算完整个 prompt
    t_prefill = time.time() - t0

    finished = torch.zeros(b, dtype=torch.bool, device=device)
    t1 = time.time()
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, top_k, top_p)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        if eos_ids is not None:
            finished |= torch.isin(nxt, eos_ids.to(nxt.device))
            if bool(finished.all()):
                break
        # decode：有 cache 只喂 1 个 token；无 cache 只能整段重算（慢，仅作对照）
        logits = model(nxt[:, None] if cache is not None else ids, cache=cache)[:, -1, :]
    t_decode = time.time() - t1

    if verbose:
        n_new = ids.shape[1] - prompt_len
        speed = n_new / t_decode if t_decode > 0 else float("inf")
        print(f"\n[性能] prefill {prompt_len} token 用时 {t_prefill * 1e3:.0f} ms；"
              f"decode {n_new} token 用时 {t_decode * 1e3:.0f} ms（{speed:.2f} token/s）")
    return ids


# =====================================================================
# 8. 自检：与 transformers 逐位对齐
# =====================================================================
def _hf_pair(cfg: Llama31Config, seed: int = 0):
    """用同一份权重造两个模型：本文件的实现 + transformers 的官方实现。"""
    from transformers import LlamaForCausalLM
    torch.manual_seed(seed)
    ours = Llama31ForCausalLM(cfg)
    hf = LlamaForCausalLM(cfg.to_hf_config())
    state = {k: v for k, v in hf.state_dict().items() if "rotary_emb" not in k}
    ours.load_state_dict(state, strict=False)
    ours.eval()
    hf.eval()
    return ours, hf


def self_check() -> None:
    torch.manual_seed(0)
    print("=" * 78)
    print("Llama 3.1 单文件推理实现 · 自检")
    print("=" * 78)

    cfg = tiny_config()

    # 1) RoPE：同一组 q/k、同一批位置，两边旋转结果一致
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding as HFRoPE, apply_rotary_pos_emb as hf_apply_rope,
    )
    ours_rope = RotaryEmbedding(cfg.head_dim, cfg.max_position_embeddings,
                                cfg.rope_theta, cfg.rope_scaling)
    hf_rope = HFRoPE(cfg.to_hf_config())
    q = torch.randn(2, cfg.num_attention_heads, 8, cfg.head_dim)
    k = torch.randn(2, cfg.num_key_value_heads, 8, cfg.head_dim)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    q_ours, k_ours = ours_rope(q, k, pos)
    cos, sin = hf_rope(q.transpose(1, 2), pos)                 # 返回 [B, S, head_dim]
    q_hf, k_hf = hf_apply_rope(q, k, cos, sin, unsqueeze_dim=1)
    d = max((q_ours - q_hf).abs().max().item(), (k_ours - k_hf).abs().max().item())
    print(f"\n[1] RoPE 旋转结果最大误差 {d:.3e}")
    assert d < 1e-6

    # 2) 整模型 logits
    ours, hf = _hf_pair(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        a = ours(ids)
        b = hf(ids).logits
    d = (a - b).abs().max().item()
    print(f"[2] logits 最大误差 {d:.3e}（相对 {d / b.abs().max().item():.2e}）")
    assert d < 1e-5

    # 3) KV Cache 增量解码 == 全量重算
    with torch.no_grad():
        logits_full = ours(ids)
        cache = ours.new_cache(batch_size=2, max_seq_len=32)
        steps = [ours(ids[:, :4], cache=cache)]         # prefill 4 个 token
        for t in range(4, 16):                           # 逐 token decode 12 步
            steps.append(ours(ids[:, t:t + 1], cache=cache))
        inc = torch.cat(steps, dim=1)
    d = (logits_full - inc).abs().max().item()
    print(f"[3] cache 增量 vs 全量 最大误差 {d:.3e}；{cache}")
    assert d < 1e-5

    # 4) greedy 生成 vs HF generate
    torch.manual_seed(1)
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    t_my = generate(ours, prompt, max_new_tokens=10, temperature=0.0)
    t_hf = hf.generate(prompt, max_new_tokens=10, do_sample=False, eos_token_id=None)
    print(f"[4] greedy 生成与 HF 一致: {torch.equal(t_my, t_hf)}  {t_my.tolist()[0]}")
    assert torch.equal(t_my, t_hf)

    # 5) 采样接口冒烟（同 seed 可复现）
    s1 = generate(ours, prompt, 8, temperature=1.0, top_k=50, top_p=0.9, seed=7)
    s2 = generate(ours, prompt, 8, temperature=1.0, top_k=50, top_p=0.9, seed=7)
    print(f"[5] 同 seed 采样可复现: {torch.equal(s1, s2)}")
    assert torch.equal(s1, s2)

    # 6) 8B 配置下的 128k 位置表
    rope8b = RotaryEmbedding(128, 131072, 500000.0, dict(ROPE_SCALING_LLAMA3))
    kv_gib = 2 * 32 * 8 * 131072 * 128 * 2 / 2**30
    print(f"[6] 8B RoPE 表 {tuple(rope8b.cos.shape)}，位置 131071 的 cos 有限 "
          f"{bool(torch.isfinite(rope8b.cos[131071]).all())}；8B@128k KV Cache ≈ {kv_gib:.1f} GiB")
    assert torch.isfinite(rope8b.cos).all()

    print("\n✅ 全部自检通过")


# =====================================================================
# 9. 运行
# =====================================================================
def resolve_model_dir(model: str) -> Path:
    """本地目录直接用；否则当作 ModelScope 模型 id 下载到缓存目录。"""
    p = Path(model)
    if p.is_dir():
        return p
    from modelscope import snapshot_download
    print(f"从 ModelScope 下载 {model} …")
    return Path(snapshot_download(model))


def eos_token_ids(tokenizer) -> list[int]:
    """收集所有结束符号：eos / <|eot_id|> / <|end_of_text|>。"""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|end_of_text|>"):
        idx = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(idx, int) and idx >= 0:
            ids.append(idx)
    return sorted(set(ids))


@torch.no_grad()
def run(args) -> None:
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_dir = resolve_model_dir(args.model)
    print(f"加载 {model_dir}（device={device}, dtype={dtype}）…")

    model = Llama31ForCausalLM.from_pretrained(model_dir, device=device, dtype=dtype)
    print(f"参数量 {model.num_parameters() / 1e9:.3f} B；层数 {model.cfg.num_hidden_layers}；"
          f"heads/kv = {model.cfg.num_attention_heads}/{model.cfg.num_key_value_heads}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    else:
        text = args.prompt
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    t0 = time.time()
    out = generate(model, input_ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                   eos_token_ids=eos_token_ids(tokenizer), verbose=args.stats)
    answer = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n>>> {args.prompt}")
    print(answer.strip())
    if not args.stats:
        print(f"\n[用时 {time.time() - t0:.2f}s · 生成 {out.shape[1] - input_ids.shape[1]} token]")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Llama 3.1 8B 从零实现（单文件推理版）")
    p.add_argument("--check", action="store_true", help="跑数值自检（不需要真实权重）")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                   help="ModelScope 模型 id 或本地 ckpt 目录")
    p.add_argument("--prompt", type=str, default="用一句话解释 RoPE 是什么。")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6, help="0 表示 greedy")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--stats", action="store_true", help="打印 prefill / decode 耗时与吞吐")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.check:
        self_check()
        return
    run(args)


if __name__ == "__main__":
    main()
