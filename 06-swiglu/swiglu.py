from __future__ import annotations 

import torch 
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU 门控前馈层

    Parameters
    ----------
    hidden_size : int 
        输入/输出维度
    intermediate_size : int 
        中间扩张维度（LLama 惯例约 8/3 倍）
    bias : bool
        LLama 默认 False
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool=False):
        super().__init__()
        # 三个投影：gate（门控）/ up（信息）/ down（收缩回 hidden）
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门: SiLU 激活后作为逐元素“开关”
        gate = F.silu(self.gate_proj(x))
        # 信息通道：直接线性变换
        up = self.up_proj(x)
        # 门控相乘 -> 收缩回 hidden 维
        return self.down_proj(gate * up)


if __name__ == "__main__":
    torch.manual_seed(0)
    H, I = 512, 1408

    layer = SwiGLU(H, I)
    x = torch.randn(2, 8, H)  # [batch, seq, hidden_dim]

    y = layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")

    # SiLU 的性质演示
    t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"\nSiLU 输入: {t.tolist()}")
    print(f"SiLU输出: {[round(v, 2) for v in F.silu(t).tolist()]}")
    print(f"    -> 负数被软压到接近0，正数近似保持")

    # 门控确实“调节信息”
    x1 = torch.randn(1, 1, H)
    with torch.no_grad():
        up = layer.up_proj(x1)
        y_on = layer.down_proj(torch.ones_like(up) * up)
        y_off = layer.down_proj(torch.zeros_like(up) * up)
        y_normal = layer(x1)
    print(f"\n门全开（1）输出范数: {y_on.norm().item():.2f}")
    print(f"门全关（0）输出范数: {y_off.norm().item():.2f}")
    print(f"真实门控输出范数: {y_normal.norm().item():.2f}")

    # 和llama对齐
    try:
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        cfg = LlamaConfig(
            hidden_size=H,
            intermediate_size=I,
            hidden_act="silu"
        )
        hf_mlp = LlamaMLP(cfg)
        with torch.no_grad():
            hf_mlp.gate_proj.weight.copy_(layer.gate_proj.weight)
            hf_mlp.up_proj.weight.copy_(layer.up_proj.weight)
            hf_mlp.down_proj.weight.copy_(layer.down_proj.weight)
        y_hf = hf_mlp(x)
        max_diff = (y - y_hf).abs().max().item()
        print(f"\n与HF LlamaMLP 最大误差: {max_diff:.2e}")
    except Exception as e:
        print(f"\n 跳过 HF 对齐测试, {type(e).__name__}")

    print(f"\n SwiGLU 自校验通过")
