"""工具注册中心：统一管理所有可供Agent 调用的工具"""
from __future__ import annotations # 延迟类型注解求值；避免潜在的循环导入

from typing import Callable, Dict # 从标准库typing模块导入两个类型注解工具：字典类型注解（Dict）和可调用函数/对象类型注解（Callable）

from .weather import get_weather
from .attraction import get_attraction

# 定义字典，并标注类型
AVAILABLE_TOOLS: Dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# 定义__all__列表，控制通配符导入的范围；例如 from tools import * ，只会导入以下三项
__all__ = ["AVAILABLE_BOOLs", "get_weather", "get_attraction"]