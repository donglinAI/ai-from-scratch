AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具：
- `get_weather(city: str)`: 查询指定城市的实时天气
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求：
你的每次回复必须严格遵循以下格式, 包含一对Thought和Action: 

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须以下之一:
1. 调用工具: function_name(arg_name="arg_value")
2. 结束任务: Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行, 不要换行
- 当收集到足够信息可以回答用户问题时, 必须使用 Action: Finish[最终答案] 格式结束

请开始吧!
"""


import requests # HTTP请求库

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息.
    """
    # API端点, 我们请求JSON格式的天气信息.
    url = f"https://wttr.in/{city}?format=j1" # Python的f-string格式, 字符串可以包含变量或表达式

    try:
        import time
        time.sleep(1)
        response = requests.get(url) # 发起网络请求
        response.raise_for_status() # 检查响应状态码为200(成功)
        data = response.json() # 解析返回的JSON数据

        # 提取当前天气状况
        current_condition = data['current_condition'][0] 
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_c']

        # 格式化自然语言返回
        return f"{city}当前天气: {weather_desc}, 气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误: 查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析数据
        return f"错误: 解析天气数据失败, 可能是城市名称无效 - {e}"

import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气, 使用Tavily Search API搜索并返回优化后的景点推荐.
    """

    # 1. 从环境变量或主程序配置中获取API密钥
    api_key = os.environ.get("TAVILY_API_KEY") # 推荐方式

    if not api_key:
        return "错误: 未配置TAVILY_API_KEY."
    
    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = f"{city}在{weather}天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API, include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净, 可以直接使用; response['answer']是一个基于所有搜索结果的总结性回答
        if response.get('answer'):
            return response['answer']
        
        # 如果没有综合性回答, 则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉, 没有找到相关的旅游景点推荐."
        
        return "根据搜索, 为您找到以下信息: \n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误: 执行Tavily搜索时出现问题 - {e}"
    

# 将所有工具函数放入一个字典, 方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端.
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应."""
        print("正在调用大语言模型...")
        try:
            import time
            time.sleep(2)
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功.")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误: 调用语言模型服务时出错."

import re

# --- 1. 配置LLM客户端 ---
API_KEY = "ms-18730a27-c29c-4a31-a1af-e28b2c65ec32" # openai
BASE_URL = "https://api-inference.modelscope.cn/v1" # openai
MODEL_ID = "Qwen/Qwen3.5-27B" # openai
os.environ["TAVILY_API_KEY"] = "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" # search

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 --- 
user_prompt = "你好, 请帮我查询一下今天北京的天气, 然后根据天气推荐一个合适的旅游景点."
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 --- 
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")

    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action; 只取第一组Thought-Action结果; 
    # 正则解释: 
    #   1. re.search(pattern, text, flags), 从全文搜索(不只从开头搜索)找到第一个匹配的片段, 找到返回match对象, 找不到返回None
    #   2. re.DOTALL: pattern需要使用到.*? 而且要求匹配到\n符; 但默认情况, . 不能匹配\n, 即到达\n则截断; 所以需要需要flags=re.DOTALL 就支持了
    #   3. .*? -> . 表示匹配任意字符; * 表示匹配0次或多次; ? 表示非贪婪模式(能不匹配就不匹配, 假设这任意字符后没有要匹配的其他字符, 那么任意字符匹配0个就行)
    #   4. () 表示捕获组, (?=) / (?:) 不是捕获组
    #   5. (Thought:.*?Action:.*?) -> 表示匹配 Thought:xxxAction:xxx的内容， 其中xx可以是任意字符或空
    #   6. ?= -> 表示中止前面捕获组; 停止方式有两个
    #   6.1. 停止方式1 -> 碰到换行符 + 任一空白符 + 任一Thought/Action/Observation
    #   6.2. 停止方式2 -> 到达字符串尾部(包括\n 会被匹配上)
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    print(prompt_history)
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation_str = "Observation: 错误: 未能解析道 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()
    print(f"action_str: {action_str}")

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务按成, 最终答案: {final_answer}")
        break
    
    # 匹配到函数名, \( 需要增加, 表示左括号(之前的那个词汇; 然后把那个词汇取出来
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    # 把左括号和右括号中间的词汇匹配出来，任何字符；取出来
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    # 正则表达式用的单引号; [^"] -> 非"的字符; 0个或多个非"的字符, 同时左右"是字面值 作为边界限定
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', action_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误: 为定义的工具 '{tool_name}'"
    
    # 3.4. 记录观察结果
    observation_str = f"observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    
