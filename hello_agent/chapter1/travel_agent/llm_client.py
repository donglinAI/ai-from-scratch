"""LLM 客户端：封装兼容 OpenAI 接口的大模型调用。"""
from __future__ import annotations

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

def _mask_api_key(api_key: str) -> str:
    """对API Key 做脱敏，仅保留前 6 位与后 4 位，便于日志排查又不泄漏。"""
    if len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


class OpenAICompatibleClient:
    """兼容 OpenAI Chat Completions 接口的 LLM 客户端。"""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        logger.info(f"LLM 客户端初始化 | model={model} | base_url={base_url} | api_key={_mask_api_key(api_key)}")

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM 生成回复。
        
        Args:
            prompt: 用户侧输入，通常为拼接后的对话历史。
            system_prompt: 系统提示词，定义 Agent 角色与输出格式。

        Returns:
            模型生成的文本；调用失败时返回错误信息字符串。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.debug(f"LLM 请求 | model={self.model} | prompt_length={len(prompt)}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )
            answer = response.choices[0].message.content or ""
            logger.info(f"LLM 响应成功 | response_length={len(answer)}")
            return answer
        except Exception as e:
            # exc_info=True 会把完整 traceback 写入日志，便于排查
            logger.error(f"LLM 调用异常 | error={e}", exc_info=True)
            return "错误：调用语言模型服务时出错。"

if __name__ == "__main__":

    # 1. 导入环境变量
    import os
    from dotenv import load_dotenv
    load_dotenv()
    print(os.environ['LLM_API_KEY'])

    # 2. 设置logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y:%m:%d %H:%M:%S"
    )

    # 3. 创建LLM服务
    llm = OpenAICompatibleClient(
        model=os.environ['LLM_MODEL_ID'],
        api_key=os.environ['LLM_API_KEY'],
        base_url=os.environ['LLM_BASE_URL']
    )
    print(llm)

    # 4. 导入系统prompt和用户prompt
    from prompts import AGENT_SYSTEM_PROMPT
    # print(AGENT_SYSTEM_PROMPT)
    prompt = "北京天气晴朗，气温27度"

    # 5. 生成结果
    result = llm.generate(prompt, AGENT_SYSTEM_PROMPT)
    logger.info(result)