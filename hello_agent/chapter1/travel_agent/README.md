# 智能旅行助手 Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OpenAI Compatible](https://img.shields.io/badge/LLM-OpenAI%20Compatible-orange.svg)](https://platform.openai.com/)

> 基于 ReAct（Reasoning + Acting）范式的智能旅行问答 Agent。通过调用天气查询与景点搜索工具，逐步推理并回答用户的旅行需求。

---

## 目录

- [功能特性](#功能特性)
- [什么是智能体 Agent](#什么是智能体-agent)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [API Key 申请指南](#api-key-申请指南)
- [快速开始](#快速开始)
- [运行示例](#运行示例)
- [技术栈](#技术栈)
- [日志说明](#日志说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 功能特性

- 🌤️ **实时天气查询**：基于 wttr.in 公开 API，支持中英文城市名
- 🏛️ **智能景点推荐**：基于 Tavily Search API，根据城市和天气条件化推荐景点
- 🤖 **ReAct 推理循环**：Thought → Action → Observation 多轮推理，自动调度工具
- 🔌 **OpenAI 兼容接口**：支持任何兼容 OpenAI Chat Completions 协议的 LLM 服务
- 🔐 **密钥安全管理**：`.env` + `python-dotenv` + `.gitignore`，源码零硬编码
- 📝 **工程化日志体系**：`logging` 统一配置，控制台 + 文件双输出，密钥自动脱敏

---

## 什么是智能体 Agent

**智能体（Agent）** 是一种能够**自主感知环境、进行推理决策、并调用工具执行动作**的 AI 系统。

传统的 LLM 调用是"一问一答"——用户输入一个问题，模型直接输出答案。而 Agent 不同，它遵循 **ReAct 范式**：

1. **Thought（思考）**：分析当前问题，决定下一步该做什么
2. **Action（行动）**：调用一个工具（如查询天气、搜索景点）
3. **Observation（观察）**：获取工具返回的结果
4. **重复 1-3**：直到收集到足够信息，输出最终答案

本项目中，Agent 的工作流程是：

```
用户："查询北京天气，然后推荐景点"
  ↓
Thought: 需要先查天气 → Action: get_weather(city="北京")
Observation: "北京当前天气: 晴, 气温27摄氏度"
  ↓
Thought: 有了天气，可以推荐景点 → Action: get_attraction(city="北京", weather="晴")
Observation: "推荐故宫、颐和园..."
  ↓
Thought: 信息足够 → Action: Finish[最终答案]
```

---

## 项目结构

```
travel_agent/
├── .env.example          # 环境变量模板
├── .gitignore            # Git 忽略规则
├── requirements.txt      # Python 依赖清单
├── README.md             # 项目说明文档
├── config.py             # 配置加载 + 全局日志初始化
├── prompts.py            # Agent 系统提示词
├── llm_client.py         # 兼容 OpenAI 接口的 LLM 客户端
├── agent.py              # ReAct Agent 主循环（解析 Thought-Action、调度工具）
├── main.py               # 程序入口
└── tools/                # 工具函数集合
    ├── __init__.py       # 工具注册中心
    ├── weather.py        # 天气查询（wttr.in）
    └── attraction.py     # 景点推荐（Tavily Search）
```

### 文件职责说明

| 文件 | 作用 |
|------|------|
| `config.py` | 从 `.env` 加载配置，初始化 `logging`（控制台+文件双输出） |
| `prompts.py` | 定义 Agent 的系统提示词，规定输出格式（Thought-Action） |
| `llm_client.py` | 封装 OpenAI 兼容客户端，日志中密钥脱敏 |
| `agent.py` | ReAct 主循环：调用 LLM → 解析 Action → 执行工具 → 记录 Observation |
| `main.py` | 组装依赖，启动 Agent |
| `tools/weather.py` | 调用 wttr.in 查询实时天气 |
| `tools/attraction.py` | 调用 Tavily Search 推荐景点 |
| `tools/__init__.py` | 用字典统一注册所有工具，Agent 通过工具名动态分发 |

---

## 环境要求

- Python 3.8+
- 可访问互联网（调用天气和搜索 API）
- 一个兼容 OpenAI 接口的 LLM 服务 API Key
- 一个 Tavily Search API Key

---

## 安装与配置

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd travel_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制模板
cp .env.example .env
```

编辑 `.env` 文件，填入你的真实 API Key：

```env
# LLM 配置（兼容 OpenAI 接口）
LLM_API_KEY=your-llm-api-key-here
LLM_BASE_URL=https://api-inference.modelscope.cn/v1
LLM_MODEL_ID=Qwen/Qwen3.5-27B

# Tavily 搜索 API
TAVILY_API_KEY=your-tavily-api-key-here

# 运行参数
MAX_ITERATIONS=5
LOG_LEVEL=INFO
```

> ⚠️ `.env` 已被 `.gitignore` 忽略，**不会被提交到 Git**，确保密钥安全。

---

## API Key 申请指南

### LLM API Key（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID）

本项目兼容任何遵循 **OpenAI Chat Completions 协议**的 LLM 服务，以下是常见选择：

| 服务 | 申请地址 | 说明 |
|------|---------|------|
| **OpenAI 官方** | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `base_url=https://api.openai.com/v1`，模型如 `gpt-4o`、`gpt-3.5-turbo` |
| **ModelScope（魔搭）** | [modelscope.cn](https://modelscope.cn/) | 阿里达摩院，有免费额度，`base_url=https://api-inference.modelscope.cn/v1`，本项目默认使用 |
| **DeepSeek** | [platform.deepseek.com](https://platform.deepseek.com/) | 深度求索，性价比高，`base_url=https://api.deepseek.com/v1`，模型 `deepseek-chat` |
| **通义千问 DashScope** | [dashscope.aliyun.com](https://dashscope.aliyun.com/) | 阿里云，需开启 OpenAI 兼容入口 |
| **智谱 AI** | [open.bigmodel.cn](https://open.bigmodel.cn/) | 智谱清言，`base_url=https://open.bigmodel.cn/api/paas/v4` |

申请步骤：
1. 注册账号并完成实名认证
2. 在控制台创建 API Key
3. 复制 Key 填入 `.env` 的 `LLM_API_KEY`
4. 对应服务的 `base_url` 和模型名填入 `LLM_BASE_URL` 和 `LLM_MODEL_ID`

### Tavily API Key（TAVILY_API_KEY）

[Tavily](https://tavily.com/) 是一个专为 AI Agent 设计的搜索 API，返回结构化的搜索结果和综合答案。

申请步骤：
1. 访问 [tavily.com](https://tavily.com/) 注册账号
2. 登录后在 Dashboard 页面找到 API Key
3. 免费额度：每月 1000 次搜索请求
4. 复制 Key 填入 `.env` 的 `TAVILY_API_KEY`

---

## 快速开始

### 运行项目

```bash
cd travel_agent
python main.py
```

### 修改用户请求

编辑 `main.py` 中的 `USER_REQUEST` 变量：

```python
USER_REQUEST = "你好, 请帮我查询一下今天杭州的天气, 然后根据天气推荐一个合适的旅游景点."
```

---

## 运行示例

```
2026-08-29 14:30:00 | INFO    | config | LLM 客户端初始化 | model=Qwen/Qwen3.5-27B | base_url=https://api-inference.modelscope.cn/v1 | api_key=ms-187...ec32
2026-08-29 14:30:00 | INFO    | agent | Agent 启动 | user_request=你好, 请帮我查询一下今天北京的天气...
2026-08-29 14:30:01 | INFO    | agent | --- 迭代 1/5 ---
2026-08-29 14:30:02 | INFO    | llm_client | LLM 响应成功 | response_length=156
2026-08-29 14:30:02 | INFO    | agent | 模型输出:
Thought: 用户需要查询北京今天的天气，我需要先调用 get_weather 工具。
Action: get_weather(city="北京")
2026-08-29 14:30:02 | INFO    | tools.weather | 调用天气查询 API | city=北京
2026-08-29 14:30:03 | INFO    | tools.weather | 天气查询成功 | 北京当前天气: Sunny, 气温27摄氏度
2026-08-29 14:30:03 | INFO    | agent | 工具返回: Observation: 北京当前天气: Sunny, 气温27摄氏度
...
==================================================
最终答案:
今天北京的天气是晴天，气温27摄氏度。根据晴朗的天气，为您推荐以下旅游景点：故宫、颐和园...
==================================================
```

日志同时输出到**控制台**和 `logs/agent.log` 文件。

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| LLM 客户端 | `openai` (Python SDK) | 调用兼容 OpenAI 接口的大模型 |
| 搜索 API | `tavily-python` | Tavily 搜索，景点推荐 |
| HTTP 请求 | `requests` | 调用 wttr.in 天气 API |
| 环境变量 | `python-dotenv` | 从 `.env` 加载配置 |
| 日志 | `logging` (标准库) | 统一日志体系 |
| Agent 范式 | ReAct (Thought-Action-Observation) | 多轮推理与工具调用 |

---

## 日志说明

项目使用 Python 标准库 `logging`，在 `config.py` 中统一配置：

- **双输出**：同时输出到控制台（`StreamHandler`）和文件（`FileHandler` → `logs/agent.log`）
- **统一格式**：`时间 | 级别 | 模块名 | 消息`，便于排查问题
- **密钥脱敏**：API Key 在日志中仅显示前 6 位和后 4 位，避免泄露
- **级别可控**：通过 `.env` 中 `LOG_LEVEL` 配置（DEBUG / INFO / WARNING / ERROR）

---

## 常见问题

### Q1: 运行时报 `ModuleNotFoundError: No module named 'dotenv'`

确保已安装依赖：`pip install -r requirements.txt`。

### Q2: 天气查询返回 `KeyError: 'temp_c'`

wttr.in API 返回的温度键是大写 `temp_C`，不是小写 `temp_c`。本项目已修复此问题。

### Q3: `logger.info` 没有输出

确保在程序入口调用了 `setup_logging()`。单独运行子模块时，需先初始化 logging。

### Q4: 可以用其他 LLM 吗？

可以。任何兼容 OpenAI Chat Completions 接口的服务都支持，只需修改 `.env` 中的 `LLM_BASE_URL` 和 `LLM_MODEL_ID`。

### Q5: 如何增加新工具？

1. 在 `tools/` 下新建 `.py` 文件，实现工具函数
2. 在 `tools/__init__.py` 的 `AVAILABLE_TOOLS` 字典中注册
3. 在 `prompts.py` 的系统提示词中说明新工具的用法
4. Agent 即可自动调用新工具

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**如果这个项目对你有帮助，欢迎给个 ⭐ Star！**

</div>
