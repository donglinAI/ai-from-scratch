"""景点推荐工具：基于 Tavily Search API。"""
from __future__ import annotations

import logging
import os

from tavily import TavilyClient

logger = logging.getLogger(__name__)

def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气搜索推荐旅游景点。
    
    Args:
        city: 城市名称。
        weather: 当前天气状况，用于条件化推荐。
    
    Returns:
        景点推荐文本；未配置密钥或搜索失败时返回错误信息。
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.error("未配置 TAVILY_API_KEY 环境变量")
        return "错误：未配置TAVILY_API_KEY."

    tavily = TavilyClient(api_key=api_key)
    query = f"{city}在{weather}天气下最值得去的旅游景点推荐及理由"
    logger.info(f"调用 Tavily 搜索 | city={city} | weather={weather}")

    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True,
        )

        # 优先使用 Tavily 基于全部结果生成的综合回答
        if response.get("answer"):
            logger.info("Tavily 返回综合回答")
            return response["answer"]

        # 无综合回答时，回退到格式化原始搜索结果
        results = response.get("results", [])
        if not results:
            logger.warning("Tavily 未返回任何搜索结果")
            return "抱歉，没有找到相关的旅游景点推荐"

        formatted = [f"- {r['title']}: {r['content']}" for r in results]
        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted)

    except Exception as e:
        logger.error(f"Tavily 搜索异常 | error={e}", exc_info=True)
        return f"错误：执行Tavily 搜索时出现错误 - {e}"
    