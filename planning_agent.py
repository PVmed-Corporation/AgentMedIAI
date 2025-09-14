from zai import ZhipuAiClient
import os
import json

SYSTEM_PROMPT = """
You are a task planner.

Your job is to take a complex user request and break it down into smaller, simple sub-queries that can be solved by existing tools.

Output Format:
- You must output a flat list of tasks (no batching).
- Each task must include:
  - "id": a unique integer identifier
  - "sub_query": the exact sub-question to solve
  - "input_path": the input path if needed, otherwise null
  - "output_path": the output path if needed, otherwise null
- If the user request is already fully solved or requires no further tasks,
  you must return:
  {
    "tasks": []
  }

The output must be valid JSON with the following structure:

{
  "tasks": [
    {
      "id": 0,
      "sub_query": "...",
      "input_path": "...",
      "output_path": "..."
    },
    {
      "id": 1,
      "sub_query": "...",
      "input_path": "...",
      "output_path": "..."
    }
  ]
}
"""


class PlanningAgent:
    def __init__(self):
        self.client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
        self.last_plan = None   # 上一次规划的结果
        self.last_result = None # 工具返回的结果

    def run(self, query: str, input_path: str, result_path: str, tool_result: dict = None) -> dict:
        """
        - 第一次调用时 tool_result=None，只做任务拆解。
        - 第二次调用时 tool_result 不为空，需要结合上次规划和工具返回结果再规划。
        """
        # 更新上次结果
        if tool_result is not None:
            self.last_result = tool_result

        # 构造 messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if self.last_plan is None:
            # 第一次调用：只做任务拆解
            messages.append({
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Input path: {input_path}\n"
                    f"Result path: {result_path}"
                ),
            })
        else:
            # 第二次调用：把上次的规划和工具结果传给模型
            messages.append({
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Previous plan: {json.dumps(self.last_plan, ensure_ascii=False, indent=2)}\n\n"
                    f"Tool result: {json.dumps(self.last_result, ensure_ascii=False, indent=2)}\n\n"
                    f"Input path: {input_path}\n"
                    f"Result path: {result_path}"
                ),
            })

        # 调用模型
        response = self.client.chat.completions.create(
            model="glm-4.5",
            temperature=0.3,
            messages=messages,
            response_format={"type": "json_object"},
        )

        # 解析 JSON
        result = json.loads(response.choices[0].message.content)

        # 保存本次规划
        self.last_plan = result

        return result