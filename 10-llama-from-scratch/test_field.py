from dataclasses import dataclass, field
ROPE_SCALING_LLAMA3 = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

# Case1：普通类（坑）
class Llama31ConfigNormal:
    rope_scaling: dict | None = ROPE_SCALING_LLAMA3

cfg1 = Llama31ConfigNormal()
cfg2 = Llama31ConfigNormal()
cfg2.rope_scaling["factor"] = 2.0
print(cfg1.rope_scaling["factor"])  # 2.0，被连带修改
print(cfg1.rope_scaling is cfg2.rope_scaling)  # True，共用同一个字典

# Case2：@dataclass 直接写dict（直接报错，禁止这种写法）
try:
    @dataclass
    class Llama31ConfigBad:
        rope_scaling: dict | None = ROPE_SCALING_LLAMA3
except ValueError as e:
    print(e)  # mutable default ... use default_factory

# Case3：@dataclass 正确写法 ✅
@dataclass
class Llama31ConfigGood:
    rope_scaling: dict | None = field(default_factory=lambda: dict(ROPE_SCALING_LLAMA3))
cfg_good1 = Llama31ConfigGood()
cfg_good2 = Llama31ConfigGood()
cfg_good2.rope_scaling["factor"] = 2.0
print(cfg_good1.rope_scaling["factor"])  # 8.0，不受影响
print(cfg_good1.rope_scaling is cfg_good2.rope_scaling)  # False，互相独立
