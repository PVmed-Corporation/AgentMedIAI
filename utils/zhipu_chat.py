"""
Zhipu Chat Unit
----------------
通用的智谱AI聊天调用封装模块，用于 glm 系列模型的统一调用。
支持结构化（JSON）与自然语言输出两种模式。
"""

import os
import json
from zai import ZhipuAiClient


class ZhipuChat:
    """统一封装 ZhipuAI Chat Completion 调用逻辑。"""

    def __init__(self, api_key: str = None, default_model: str = "glm-4.6", default_temp: float = 1.0):
        self.client = ZhipuAiClient(api_key=(api_key or os.environ.get("API_KEY")))
        self.default_model = default_model
        self.default_temp = default_temp

    # ---------- 结构化输出（JSON） ----------
    def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        temperature: float = None,
        response_format: str = "json_object",
        thinking: dict = None,
    ) -> dict:
        """
        用于需要结构化响应的任务（例如 tool_call_agent、自动规划）。

        参数:
            system_prompt: 系统提示词
            user_prompt: 用户输入
            model: 使用的模型名称（默认 glm-4.6）
            temperature: 控制输出随机性
            response_format: 响应格式（默认 json_object）
            thinking: 控制思维链，可选值 {"type": "enabled"} 或 {"type": "disabled"}
        """
        model = model or self.default_model
        temperature = temperature or self.default_temp

        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": response_format},
            thinking=thinking or {"type": "disabled"},  # ✅ 支持思维链参数
        )

        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
            return {"raw_content": response.choices[0].message.content}

    # ---------- 自然语言输出 ----------
    def ask_text(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        system_prompt: str = None,
        thinking: dict = None,
    ) -> str:
        """
        用于自然语言对话、总结或报告生成。

        参数:
            prompt: 用户输入
            model: 使用的模型名称（默认 glm-4.6）
            temperature: 控制输出随机性
            system_prompt: 系统提示词（可选）
            thinking: 控制思维链，可选值 {"type": "enabled"} 或 {"type": "disabled"}
        """
        model = model or self.default_model
        temperature = temperature or self.default_temp
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            thinking=thinking or {"type": "disabled"},  # ✅ 支持思维链参数
        )

        return response.choices[0].message.content


# ---------- 模块独立调试 ----------
if __name__ == "__main__":
    chat = ZhipuChat()

    # 示例：结构化输出
    query = "Use CXAS to extract Cardio-Thoracic Ratio (CTR)"
    tools_prompt = {"tool_name": "CXAS"}
    system_prompt = "You are an expert assistant."
    user_prompt = f"User query: {query}\nAvailable tools:\n{json.dumps(tools_prompt, indent=2)}"

    print("\n--- JSON 模式 ---")
    result_json = chat.ask_json(system_prompt, user_prompt, thinking={"type": "enabled"})  # ✅ 开启思维链
    print(result_json)

    # 示例：自然语言输出
    print("\n--- 文本模式 ---")
    summary_prompt = "Please summarize the following tool usage results:\n" + json.dumps(result_json, indent=2)
    result_text = chat.ask_text(summary_prompt, system_prompt="You are a summarization expert.", thinking={"type": "enabled"})
    print(result_text)
