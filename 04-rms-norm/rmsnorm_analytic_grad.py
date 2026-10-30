import torch

def rmsnorm_forward(a, g, eps=1e-6):
    """
    RMSNorm + per-element gain 前向
    Args:
        a: (..., n) 输入向量，最后一维做归一化
        g: (n,) 可学习逐元素增益
        eps: 数值稳定项
    Returns:
        v: 激活输入，(..., n)
        cache: 用于反向的缓存 (a, abar, rms, g)
    """
    rms = torch.sqrt(a.pow(2).mean(dim=-1, keepdim=True) + eps)
    abar = a / rms
    v = abar * g
    return v, (a, abar, rms, g)


def rmsnorm_backward(dv, cache):
    """
    RMSNorm + gain 反向，返回 grad_a, grad_g
    Args:
        dv: ∂L/∂v, (..., n) 上游梯度
        cache: 前向缓存
    Returns:
        grad_a: ∂L/∂a, (..., n)
        grad_g: ∂L/∂g, (n,) 沿所有批量维度规约求和
    """
    a, abar, rms, g = cache
    n = a.shape[-1]
    d = g * dv  # d = g ⊙ ∂L/∂v
    inner = (abar * d).sum(dim=-1, keepdim=True) / n
    grad_a = (d - abar * inner) / rms
    grad_g = (dv * abar).reshape(-1, n).sum(dim=0)
    return grad_a, grad_g


def linear_backward(grad_a, x):
    """
    线性层反向：a = W @ x（batch 版 a = x @ W.T），∂L/∂W = grad_a^T @ x
    这里约定：x: (..., m), a: (..., n)，W: (n, m)
    grad_a: ∂L/∂a, (..., n)
    x: (..., m)
    return ∂L/∂W: (n, m)
    """
    return torch.einsum("...n,...m->nm", grad_a, x)


if __name__ == "__main__":
    # ========== 测试用例：和论文附录数值完全对齐 ==========
    # 配置：a = [3,4], g=[1,1], dv = ∂L/∂v = [1,1], x=[1,2]
    torch.manual_seed(42)
    a = torch.tensor([[3.0, 4.0]])       # shape (1, 2)
    g = torch.tensor([1.0, 1.0])         # shape (2,)
    dv = torch.tensor([[1.0, 1.0]])      # ∂L/∂v
    x = torch.tensor([[1.0, 2.0]])       # shape (1, 2)

    # 前向
    v, cache = rmsnorm_forward(a, g)
    a_cached, abar, rms, g_cached = cache
    print("===== 前向结果 =====")
    print(f"a         = {a.squeeze()}")
    print(f"rms       = {rms.item():.4f}")
    print(f"abar      = {abar.squeeze()}")
    print(f"v         = {v.squeeze()}")

    # RMSNorm反向
    grad_a, grad_g = rmsnorm_backward(dv, cache)
    print("\n===== RMSNorm 反向梯度 =====")
    print(f"∂L/∂g     = {grad_g}")
    print(f"∂L/∂a     = {grad_a.squeeze()}")

    # 线性层反向求 ∂L/∂W
    grad_W = linear_backward(grad_a, x)
    print("\n===== 线性层权重梯度 ∂L/∂W =====")
    print(grad_W)

    # vec(∂L/∂W) 列主序拉直
    vec_gradW = grad_W.reshape(-1)
    print(f"\n===== vec(∂L/∂W) (列主序) =====")
    print(vec_gradW)

    # ========== 自动微分校验（torch.autograd） ==========
    print("\n===== Autograd 校验（验证解析梯度正确性） =====")
    a_autograd = torch.tensor([[3.0, 4.0]], requires_grad=True)
    g_autograd = torch.tensor([1.0, 1.0], requires_grad=True)
    v_autograd, _ = rmsnorm_forward(a_autograd, g_autograd)
    loss = (v_autograd * torch.tensor([[1.,1.]])).sum()  # 等价 dv = [1,1]
    loss.backward()
    print(f"autograd ∂L/∂a = {a_autograd.grad.squeeze()}")
    print(f"autograd ∂L/∂g = {g_autograd.grad}")
