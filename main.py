import concurrent.futures
from planning_agent import PlanningAgent
from tool_call_agent import ToolCallAgent
from tool_invoke import ToolInvoker
from zai import ZhipuAiClient
from pathlib import Path
import json
import os

def run_subtask(task, tools_path, max_retries=5):
    tool_call_agent = ToolCallAgent()
    tool_invoker = ToolInvoker()
    sub_query = task["sub_query"]
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    attempt = 0
    result = None
    tool_call_json = None
    while attempt < max_retries:
        tool_call_json = tool_call_agent.run(sub_query, tools_path, input_path, output_path)
        result = tool_invoker.invoke(tool_call_json)
        if result.get("status") == "success":
            break
        attempt += 1
        # 失败时将错误信息反馈给 ToolCallAgent（可扩展为更智能的提示）
        sub_query += f"\nTool invocation error: {result.get('stderr', '')}"
    return {
        "tool_name": tool_call_json.get("tool_name"),
        "command": tool_call_json.get("command"),
        "result": result,
        "sub_query": sub_query,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "attempts": attempt + 1
    }

def main():
    tools_path = "tools.json"
    input_path = "/data/result/yilinyou/00000001_000.png"
    result_path = "/data/result/yilinyou/CXAS_result/"
    query = input("请输入指令：")
    planning_agent = PlanningAgent()
    plan = planning_agent.run(query, input_path, result_path)
    tasks = plan.get("tasks", [])
    results = []

    # 并行执行所有子任务
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_subtask, task, tools_path)
            for task in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 汇总所有工具、命令、结果，生成自然语言描述
    summary = "工具调用结果如下：\n"
    for r in results:
        summary += (
            f"子任务: {r['sub_query']}\n"
            f"工具: {r['tool_name']}\n"
            f"命令: {r['command']}\n"
            f"结果: {json.dumps(r['result'], ensure_ascii=False)}\n"
            f"尝试次数: {r['attempts']}\n"
            "----------------------\n"
        )

    # 用 GLM4.5 总结
    client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
    response = client.chat.completions.create(
        model="glm-4.5",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "请用自然语言总结以下工具调用过程和结果。"},
            {"role": "user", "content": summary}
        ]
    )
    print("=== 汇总结果 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
