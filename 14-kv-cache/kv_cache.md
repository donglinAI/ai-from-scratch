# KV Cache 缓存机制详解

## 0. 一句话结论

**KV Cache 把每一层已经算过的 Key / Value 向量缓存下来，让自回归生成每一步只为「新 token」计算一次注意力，而不是对整段历史反复重算。** 它把逐 token 生成的计算量从「每步 $`O(\text{序列}^2)`$」降到「每步 $`O(\text{序列})`$」，是长文本推理能落地的前提。

---

## 1. 什么是 KV Cache

自回归语言模型逐 token 生成，每生成一个新 token，都要拿它的 Query 与**所有历史 token** 的 Key、Value 计算注意力：

```math
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
```

关键观察：第 $`t`$ 步计算注意力时，历史 token 的 $`K`$、$`V`$ 和上一步（第 $`t-1`$ 步）计算时**完全一样**——因为它们是同一个 token 在同一层的投影结果。既然如此，就只算一次、存下来复用，这就是 KV Cache。

用你代码里的话说：`KVCache` 是一个「预分配静态表」，`update` 负责「写入切片、返回读取视图」，后续步骤直接复用，不再重复投影。

---

## 2. 为什么需要：消除 $`O(N^2)`$ 级别的冗余计算

设 prompt 长 $`P`$，要生成 $`N`$ 个新 token。

**不用 KV Cache**（每步都对全序列重算）：

- 第 1 步：对 $`P`$ 个 token 算 attention，$`K/V\in\mathbb{R}^{P\times d}`$
- 第 2 步：对 $`P+1`$ 个 token 重算，$`K/V\in\mathbb{R}^{(P+1)\times d}`$
- ……
- 第 $`N`$ 步：对 $`P+N`$ 个 token 重算

每一步的注意力打分矩阵都是 $`(\text{当前长度})^2`$，总计算量：

```math
\sum_{t=1}^{N}(P+t)^2\;\sim\;O(N^3)
```

**用 KV Cache**（历史 $`K/V`$ 只算一次）：

- Prefill 一次算完 $`P`$ 个 token 的 $`K/V`$ 并缓存
- 之后每步只算新 token 的一个 $`Q/K/V`$，注意力只对「历史 + 当前」做一次线性长度的打分：

```math
\sum_{t=1}^{N}(P+t)\;\sim\;O(N^2)
```

量级差异是「每步 $`O(\text{序列}^2)`$ → 每步 $`O(\text{序列})`$」。没有 KV Cache，8B 模型生成 1000 个 token 的注意力计算量会大到不可用。

---

## 3. 两阶段：Prefill 与 Decode

| 阶段 | 输入 | 计算特征 | 瓶颈 |
| --- | --- | --- | --- |
| **Prefill**（预填充） | 整个 prompt | 一次性算完所有 token 的 $`Q/K/V`$，$`K/V`$ 全部写入 cache，取最后一个位置的 logits 采样第一个新 token | 计算密集（吃 GPU 算力） |
| **Decode**（增量解码） | 每次 1 个新 token | 只算新 token 的 $`Q/K/V`$，$`K/V`$ 追加进 cache，与历史做注意力 | 访存密集（吃显存带宽） |

你的代码里这个两阶段在 `generate()` 中清晰对应：

```python
logits = model(ids, cache=cache)[:, -1, :]  # prefill：一次喂完整个 prompt
# ...
logits = model(nxt[:, None], cache=cache)[:, -1, :]  # decode：每次只喂 1 个 token
```

Prefill 之所以能「一次算完」，是因为 prompt 的所有 token 都已就绪，可并行；Decode 之所以「每次只喂 1 个」，是因为下一个 token 依赖当前采样结果，只能串行。

---

## 4. 内存占用公式

KV Cache 的显存公式：

```math
\operatorname{bytes}=2\times L\times H_{kv}\times S_{\max}\times d_{\mathrm{head}}\times b_{\mathrm{elem}}
```

- 系数 $`2`$：$`K`$ 和 $`V`$ 两份
- $`L`$：层数
- $`H_{kv}`$：**kv 头数**（不是 q 头数，这是 GQA 的关键）
- $`S_{\max}`$：最大序列长度
- $`d_{\mathrm{head}}`$：每头维度
- $`b_{\mathrm{elem}}`$：单个元素的字节数，对应代码中的 `bytes_per_elem`；bf16 为 2 字节，fp32 为 4 字节

**Llama 3.1 8B @ 128k 上下文的例子**（正是你代码注释里的计算）：

```math
2\times 32\times 8\times 131072\times 128\times 2\ \operatorname{bytes}
\approx 17.2\ \operatorname{GB}
```

注意：8B 权重本身 bf16 只有 16 GB，KV Cache（17.2 GB）**比权重还大**。这是长上下文推理最主要的显存消耗，也是量化 KV Cache（如 int8/int4 KV）技术存在的动机。

---

## 5. GQA 如何缓解 KV Cache 压力

Llama 3.1 8B：32 个 q 头，但只有 8 个 kv 头（GQA，Grouped-Query Attention），$`n_{\mathrm{rep}}=32/8=4`$。

- 若用 MHA（32 q = 32 kv）：$`H_{kv}=32`$，KV Cache 是现在的 4 倍 ≈ 68.8 GB，128k 上下文直接爆显存
- GQA 让 4 个 q 头共享同一组 kv 头，KV Cache 缩小 4 倍

这就是你代码里 `repeat_kv` 的意义：把 8 个 kv 头用 `expand` 扩成 32 个 q 头视角，**只建视图、不拷贝数据**：

```python
def repeat_kv(x, n_rep):
    b, kv_heads, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, kv_heads, n_rep, s, d)
    return x.reshape(b, kv_heads * n_rep, s, d)
```

---

## 6. 代码讲解：`KVCache` 类的三个设计

```python
class KVCache:
    def __init__(self, num_layers, batch_size, num_kv_heads, max_seq_len, head_dim, ...):
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, ...) for _ in range(num_layers)]  # 预分配
        self.v = [torch.zeros(shape, ...) for _ in range(num_layers)]
        self.seq_len = 0

    def update(self, layer_idx, k, v):
        end = self.seq_len + k.shape[2]  # 本步写到哪
        self.k[layer_idx][:, :, self.seq_len:end] = k  # 切片写入
        self.v[layer_idx][:, :, self.seq_len:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]  # 读取视图

    def advance(self, n=1):
        self.seq_len += n  # 指针前移
```

| 设计 | 动机 |
| --- | --- |
| **预分配静态表** | 一次性 `torch.zeros` 分满 `max_seq_len`，避免动态 `cat` 反复触发显存重分配（这在大模型推理里是常见的性能杀手） |
| **写入切片、读取视图** | `update` 返回的 `[:, :, :end]` 是零拷贝视图，「历史 + 当前」直接拼出来，不产生新显存 |
| **`seq_len` 指针** | 记录已填充长度，它同时是「下一步写入位置」和「RoPE 的起始相位」 |

---

## 7. 因果 Mask 的构造（Decode 时最容易写错的地方）

Decode 阶段，当前 token 的 query 绝对位置是 `past + i`，只能看到 `j <= past + i` 的 key：

```python
q_pos = torch.arange(past, past + s)  # query 的绝对位置
k_pos = torch.arange(kv_len)  # key 的绝对位置
allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
mask.masked_fill_(~allowed, -inf)
```

**mask 必须基于绝对位置**（`past + i`），而不是本步的相对位置 `[0, s)`。否则 Decode 阶段的新 token 会错误地「看不到历史」，输出完全错乱。

---

## 8. 关键陷阱清单

1. **RoPE 位置从 `past` 接着数**：Decode 时 `position_ids` 必须以 `past` 为起点，否则旋转相位错位。
2. **softmax 抬到 fp32**：bf16 下直接求 softmax 会累积误差（代码里 `F.softmax(scores, dim=-1, dtype=torch.float32)`）。
3. **容量检查**：`end > max_seq_len` 要显式报错，不能静默溢出覆盖已有缓存。
4. **prefill 也要写 cache**：很多人只记得 decode 写 cache，漏了 prefill 阶段的 $`K/V`$ 也必须缓存，否则后续 decode 没有历史可看。
