"""Byte Pair Encoding (BPE) 分词器

核心思路（三步）：
1. 初始词表 = 256 个单字节 (0..255)，每个 token 就是 1 个字节。
2. 训练：反复统计语料中相邻 token 对的出现频率，把最高频的一对合并成一个
   新 token，直到词表达到目标大小。合并的先后顺序本身即"优先级"——
   越早合并的 pair 优先级越高（记录在 merges 里）。
3. 编码：文本转字节序列，按 merges 的优先级贪心合并（每次挑"可合并且优先级
   最高"的 pair 先合）；解码：把 token id 逐个映射回字节串再拼接。

说明：
- 这是"最小可用"的 BPE，未做 GPT-2 那种基于正则的预分词（regex pre-tokenization），
  也不含 <|endoftext|> 等特殊 token，目的是把核心算法看清楚。真实词表在此基础上加即可。
- 训练与编码共用 _get_stats / _merge 两个原语，逻辑高度对称。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _get_stats(ids: list[int]) -> Counter:
    """统计序列里所有相邻 token 对 (a, b) 的出现次数。"""
    return Counter(zip(ids, ids[1:]))


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """把序列里所有相邻的 pair 替换成新 token idx（单趟、不重叠合并）。"""
    out: list[int] = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        # pair -> 新 token id；dict 的插入顺序即合并优先级（越早值越小）
        self.merges: dict[tuple[int, int], int] = {}
        # token id -> 字节串
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------ 训练 ------------------------------
    def train(self, texts, vocab_size: int, verbose: bool = False) -> None:
        """在语料上训练 BPE，直到词表达到 vocab_size。texts 可为 str 或 list[str]。"""
        if isinstance(texts, str):
            texts = [texts]
        ids = list("\n".join(texts).encode("utf-8"))
        num_merges = vocab_size - 256
        assert num_merges >= 0, "vocab_size 必须 >= 256"

        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)      # 取频率最高的 pair
            idx = 256 + i
            ids = _merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

        if verbose:
            print(f"[BPE] {len(self.merges)} 次合并，词表大小 {self.vocab_size}")

    # ---------------------------- 编码 / 解码 ----------------------------
    def encode(self, text: str) -> list[int]:
        """文本 -> token id 列表。按 merges 优先级贪心合并。"""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # 在"当前可合并的 pair"里，挑 merges 中优先级最高（值最小）的
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """token id 列表 -> 文本。"""
        parts = [self.vocab[i] for i in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------------------- 保存 / 加载 ----------------------------
    def save(self, path: str | Path) -> None:
        # merges 存成有序列表，顺序即优先级；load 时按 256+i 重建 id
        data = {
            "merges": [[a, b] for (a, b) in self.merges],
            "vocab": {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok.vocab = {int(k): v.encode("utf-8") for k, v in data["vocab"].items()}
        for i, (a, b) in enumerate(data["merges"]):
            tok.merges[(a, b)] = 256 + i
        return tok


if __name__ == "__main__":
    # 自检 1：训练 -> 编码 -> 解码 往返一致
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the dog barks and the fox runs away",
        "a quick brown fox and a lazy dog are friends",
    ] * 20

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=300, verbose=True)

    text = "the quick brown fox jumps over the lazy dog"
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "往返解码不一致"
    print(f"[1] 往返一致：'{text}' -> {len(ids)} tokens -> '{tok.decode(ids)}'")

    # 自检 2：训练后词表应比原始文本更"紧凑"（token 数 < 字节数）
    raw_bytes = len(text.encode("utf-8"))
    print(f"[2] 原始字节数 {raw_bytes}，BPE 后 token 数 {len(ids)}"
          f"（压缩比 {len(ids) / raw_bytes:.2f}）")

    # 自检 3：save / load 往返一致
    tok.save("/tmp/_bpe_test.json")
    tok2 = BPETokenizer.load("/tmp/_bpe_test.json")
    assert tok2.decode(tok2.encode(text)) == text and tok2.vocab_size == tok.vocab_size
    print("[3] save/load 往返一致")

    print("\n✅ BPE 自检通过")
