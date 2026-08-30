"""ReAct Agent 主循环：解析 Thought-Action，调度工具，直至 Finish。"""
from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Optional

from llm_client import OpenAICompatibleClient
from prompts import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# --- 正则预编译（模块加载时编译一次，避免循环内重复编译 ） ---

# 提取第一组 Thought-Action: 遇到下一组Thought/Action/Observation 或字符串结尾时停止
_THOUGHT_ACTION_RE = re.compile(r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)", re.DOTALL)
# 从模型输出中提取 Action 行及后续内容
_ACTION_RE = re.compile(r"Action: (.*)", re.DOTALL)
# 匹配 Finish[答案]
_FINISH_RE = re.compile(r"Finish\[(.*)\]", re.DOTALL)
# 匹配 函数名（参数）形式的工具调用
_TOOL_CALL_RE = re.compile(r"(\w+)\((.*)\)", re.DOTALL)
# 从工具参数字符串中提取 key="value" 形式的命名参数
_KWARGS_RE = re.compile(r'(\w+)="([^"]*)"')


class TravelAgent:
    """旅行助手 Agent: 基于 ReAct 范式循环推理与工具调用。"""

    def __init__(
            self,
            llm: OpenAICompatibleClient,
            tools: Dict[str, Callable[..., str]],
            max_iterations: int=5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.history: list[str] = []

    def run(self, user_request: str) -> str:
        """执行 Agent 主循环，返回最终答案。
        
        Args:
            user_request: 用户的原始旅行请求。

        Returns:
            Agent 给出的最终答案；达到最大轮次仍未Finish 时返回错误信息。
        """
        self.history = [f"用户请求：{user_request}"]
        logger.info(f"Agent 启动 | user_request={user_request}")

        for i in range(self.max_iterations):
            logger.info(f"--- 迭代 {i+1}/{self.max_iterations} ---")

            # 1. 调用 LLM 生成 Thought-Action
            llm_output = self._call_llm()
            llm_output = self._truncate_thought_action(llm_output)
            self.history.append(llm_output)
            logger.info(f"模型输出:\n{llm_output}")

            # 2. 解析并执行 Action；返回非 None 表示任务已 Finish
            final_answer = self._parse_and_execute(llm_output)
            if final_answer is not None:
                logger.info(f"Agent 完成 | 最终答案: {final_answer}")
                return final_answer

        logger.warning(f"达到最大迭代次数 {self.max_iterations}, 未获得 Finish")
        return "错误：已达到最大思考轮次，未能得出最终答案。"

    def _call_llm(self) -> str:
        """将对话历史拼接为 prompt 并调用 LLM。"""
        full_prompt = "\n".join(self.history)
        return self.llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

    @staticmethod
    def _truncate_thought_action(text: str) -> str:
        """仅保留第一组 Thought-Action, 截断模型多余输出。 
        模型有时会一次性输出多组 Thought-Action, Agent 每轮只应执行一组，
        因此截取第一组后把剩余部分丢弃，避免历史被污染。
        """
        match = _THOUGHT_ACTION_RE.search(text)
        if match:
            truncated = match.group(1).strip()
            if truncated != text.strip():
                logger.debug("已截取多余的 Thought-Action")
                return truncated
        return text

    def _parse_and_execute(self, llm_output: str) -> Optional[str]:
        """解析 Action 并执行对应操作。
        
        Returns:
            Finish 时返回最终答案；需要继续循环时返回None。
        """
        action_match = _ACTION_RE.search(llm_output)
        if not action_match:
            observation = (
                "Observation: 错误：未能解析到 Action 字段。 请确保回复严格遵循 'Thought: ... Action: ...' 格式"
            )
            logger.warning("未解析到 Action 字段")
            self.history.append(observation)
            return None
        
        action_str = action_match.group(1).strip()
        logger.info(f"解析到 Action: {action_str}")

        # --- 分支 1: 结束任务 --- 
        if action_str.startswith("Finish"):
            finish_match = _FINISH_RE.match(action_str)
            if finish_match:
                return finish_match.group(1)
            logger.error(f"Finish 格式错误： {action_str}")
            return "错误：Finish 格式不正确，应为 Finish[答案]"
        
        # --- 分支 2: 工具调用 --- 
        tool_match = _TOOL_CALL_RE.match(action_str)
        if not tool_match:
            observation = f"Observation：错误：无法解析工具调用 - {action_str}"
            logger.warning(f"工具调用解析失败：{action_str}")
            self.history.append(observation)
            return None

        tool_name = tool_match.group(1)
        kwargs = dict(_KWARGS_RE.findall(action_str))

        if tool_name not in self.tools:
            observation = f"Observation: 错误：未定义的工具 '{tool_name}'"
            logger.warning(f"调用未定义工具: {tool_name}")
            self.history.append(observation)
            return None 
        
        logger.info(f"执行工具 | name={tool_name} | kwargs={kwargs}")
        observation = self.tools[tool_name](**kwargs)
        observation_str = f"Observation: {observation}"
        logger.info(f"工具返回：{observation_str}")
        self.history.append(observation_str)
        return None