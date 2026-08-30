"""应用配置: 从环境变量加载所有敏感配置与运行参数。"""
from __future__ import annotations # 有了该行，即使类型注释不存在也不会报错，例如 func(name: str) -> strxx, 

import logging
import os
from dataclasses import dataclass 

from dotenv import load_dotenv # dotenv作用: 把.env文件中的key-value注入到environ环境变量中

# 加载项目根目录下的.env文件（若存在）
load_dotenv() # 如果目录下没有.env文件 也不会报错；但在后续os.environ取值时会报错

@dataclass(frozen=True)
class LLMConfig:
    """LLM 服务连接配置。"""
    api_key: str
    base_url: str
    model_id: str

@dataclass(frozen=True)
class Config:
    """全局运行配置聚合。"""
    llm: LLMConfig
    tavily_api_key: str
    max_iterations: int = 5
    log_level: str = "INFO"


def _require_env(name: str) -> str:
    """读取必填环境变量，缺失时抛出带修复指引的异常。"""
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"缺少必填环境变量：{name}。请判断项目目录是否有.env文件，如果有，进一步判断.env文件中是否包含变量：{name}"
        )
    return value

def load_config() -> Config:
    """从环境变量构建不可变配置对象。"""
    llm = LLMConfig(
        api_key=_require_env("LLM_API_KEY"),
        base_url=_require_env("LLM_BASE_URL"),
        model_id=_require_env("LLM_MODEL_ID")
    )
    return Config(
        llm=llm,
        tavily_api_key=_require_env("TAVILY_API_KEY"),
        max_iterations=int(os.environ.get("MAX_ITERATIONS", "5")),
        log_level=os.environ.get("LOG_LEVEL", "INFO")
    )

def setup_logging(level: str = "INFO") -> None:
    """配置全局日志：同时输出到控制台与文件，带统一格式。
    
    日志文件位于 logs/agent.log；采用 UTF-8 编码避免中文乱码

    测试：python -c "from config import load_config, setup_logging; setup_logging(); import logging; logging.getLogger('test').info('hello')"
    """
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/agent.log", encoding="utf-8")
        ]
    )

