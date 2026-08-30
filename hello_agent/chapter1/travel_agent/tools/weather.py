"""天气查询工具：基于 wttr.in 公开API。"""
from __future__ import annotations
import logging
import requests

logger = logging.getLogger(__name__)

_WTTR_URL = "https://wttr.in/{city}?format=j1"
_TIMEOUT_SECONDS = 10

import time
from functools import wraps

def rate_limit(second=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(second)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_weather(city: str) -> str:
    """查询指定城市的实时天气

    Args:
        city: 城市名称，支持中英文
    
    Returns:
        格式化的天气描述字符串；查询或解析失败时返回错误信息。
    """
    url = _WTTR_URL.format(city=city)
    logger.info(f"调用天气查询 API: city={city}") # logging.basicConfig() 在main.py函数中设置

    try:
        response = requests.get(url, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]

        result = f"{city}当前天气: {weather_desc}, 气温{temp_c}摄氏度"
        logger.info(f"天气查询成功 | {result}")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"天气查询网络错误 | city={city} | error={e}")
        return f"错误: 查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        logger.error(f"天气数据解析失败 | city={city} | error={e}")
        return f"错误: 解析天气数据失败，可能是城市名称无效 - {e}"

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y:%m:%d %H:%M:%S"
    )
    get_weather("beijing")