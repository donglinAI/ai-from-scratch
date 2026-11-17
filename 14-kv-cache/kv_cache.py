"""KV Cache 最小实现（PyTorch）
核心内容
    预分配静态表 -> prefill 写入 -> decode 逐 token 追加 -> 复用历史 K/V

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """最小 KV Cache：预分配 [B, num_kv_heads, max_seq_len, head_dim] 静态表。

    三个要点（与完整版一致）：
    1. 预分配：一次 torch.zeros 分满 max_seq_len，避免动态 cat 反复触发显存重分配
    2. 写入切片 / 读取视图：update 返回 [:, :, :end] 零拷贝视图，历史+当前直接拼出
    3. seq_len 指针：记录已填充长度，它同时是下一步写入位置和 RoPE 起始相位
    """

    def __init__(self, num_layers, batch_size, num_kv_heads, max_seq_len, head_dim):
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = [torch.zeros(shape) for _ in range(num_layers)]
        self.v = [torch.zeros(shape) for _ in range(num_layers)]
        self.seq_len = 0

    def update(self, layer_idx, k, v):
        """把本步的 k/v 写到 cache，返回"历史 + 当前"视图（不拷贝）。"""
        end = self.seq_len + k.shape[2]
        self.k[layer_idx][:, :, self.seq_len:end] = k
        self.v[layer_idx][:, :, self.seq_len:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, n=1):
        self.seq_len += n


class Attention(nn.Module):
    """带 KV Cache 的因果自注意力（单层 MHA，无 RoPE，专为演示缓存机制）。

    注意：这里为了聚焦缓存机制，简化成 MHA（num_heads 个 kv 头），
    且 forward 不自动 advance——由调用方显式推进，语义更直白。
    """

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)

    def forward(self, x, cache=None, layer_idx=0):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # 关键：past = 已缓存长度；有 cache 就复用，无 cache 时 kv 就是本步全部
        past = cache.seq_len if cache is not None else 0
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)   # [B, H, past+s, D]
        kv_len = past + s

        # 因果 mask 必须基于绝对位置：query 位置 = past + i，只能看 j <= past + i
        q_pos = torch.arange(past, past + s)
        k_pos = torch.arange(kv_len)
        causal = k_pos[None, :] <= q_pos[:, None]           # [s, kv_len]，True=允许看

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, s, kv_len]
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)                     # 生产代码建议抬到 fp32
        out = (attn @ v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)


def demo():
    """自检：KV Cache 增量解码 == 一次性全量重算（数值应完全一致）。"""
    torch.manual_seed(0)
    hidden, heads, head_dim = 32, 4, 8
    attn = Attention(hidden, heads, head_dim)
    cache = KVCache(num_layers=1, batch_size=1, num_kv_heads=heads,
                    max_seq_len=64, head_dim=head_dim)

    x = torch.randn(1, 16, hidden)   # 16 个 token 的 hidden state

    # 对照组：一次性算完 16 个 token（无 cache）
    full = attn(x)

    # 实验组：prefill 4 个 -> 逐 token decode 12 步（有 cache）
    pieces = [attn(x[:, :4], cache, layer_idx=0)]
    cache.advance(4)
    for t in range(4, 16):
        pieces.append(attn(x[:, t:t + 1], cache, layer_idx=0))
        cache.advance(1)
    inc = torch.cat(pieces, dim=1)

    diff = (full - inc).abs().max().item()
    print(f"[自检] KV Cache 增量解码 vs 全量重算，最大误差 = {diff:.2e}")
    assert diff < 1e-5, "数值不一致：检查 mask / cache 指针"
    print("[自检] 通过 —— 缓存 K/V 后复用，结果与全量重算逐位一致")


if __name__ == "__main__":
    demo()
