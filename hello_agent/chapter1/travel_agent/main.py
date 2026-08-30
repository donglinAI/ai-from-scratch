"""旅行助手 Agent 程序入口"""
from __future__ import annotations

import logging
from agent import TravelAgent
from config import load_config, setup_logging
from llm_client import OpenAICompatibleClient
from tools import AVAILABLE_TOOLS

logger = logging.getLogger(__name__)

# 示例用户请求；实际使用时可改为接受命令行参数或用户输入
USER_REQUEST = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"


def main() -> None:
    """加载配置、初始化组件并运行 Agent。"""
    config = load_config()
    setup_logging(config.log_level)

    llm = OpenAICompatibleClient(
        model=config.llm.model_id,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )

    agent = TravelAgent(
        llm=llm,
        tools=AVAILABLE_TOOLS,
        max_iterations=config.max_iterations
    )

    answer = agent.run(USER_REQUEST)

    print("\n" + "=" * 50)
    print(f"最终答案：{answer}\n" + "="*50)

if __name__ == "__main__":
    main()