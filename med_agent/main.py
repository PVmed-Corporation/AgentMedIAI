import concurrent.futures
from planning_agent import PlanningAgent
from tool_call_agent import ToolCallAgent
from tool_invoke import ToolInvoker
from zai import ZhipuAiClient
from pathlib import Path
import json
import os
from generate_tool_info import build_tools_prompt
from retriever import ToolRetriever

def run_subtask(task, tools_path, max_retries=4):
    tool_call_agent = ToolCallAgent()
    tool_invoker = ToolInvoker()
    sub_query = task["sub_query"]
    #有些任务不需要输入输出路径
    input_path = Path(task["input_path"]) if task.get("input_path") else None
    output_path = Path(task["output_path"]) if task.get("output_path") else None
    attempt = 0
    result = None
    tool_call_json = None
    error_msg = ""
    #如果工具数量比较少就不适用retriever
    """
    tool_call_json schema:
    {
      "tool_name
      "explanation"
      "command_type"
      "command"
    }
    """

    retriever = ToolRetriever()
    selected = retriever.prompt_based_retrieval(sub_query, tools_path)
    tools_prompt = build_tools_prompt(selected, "tool")

    while attempt < max_retries:
        tool_call_json = tool_call_agent.run(sub_query, input_path, output_path,tools_prompt, bug=(attempt > 0), error_msg=error_msg)
        print("********command**********")
        print(json.dumps(tool_call_json, indent=4))
        result = tool_invoker.invoke(tool_call_json)
        if result.get("status") == "success":
            break
        attempt += 1
        # 失败时将错误信息反馈给 ToolCallAgent（可扩展为更智能的提示:日志系统）
        error_msg = (result.get("stdout", "") or "").strip()
        if not error_msg:
            error_msg = "Unknown error (no output captured)."
        print(error_msg)
        if attempt < max_retries:
            print("retrying...")
        else:
            print("max retries reached, moving on...")
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
    input_path = "/data/result/zhangjie/image/ETT.jpg"
    result_path = "/data/result/yilinyou/CXAS_result/"
    query = input("请输入指令：")
    planning_agent = PlanningAgent()
    plan = planning_agent.run(query, input_path, result_path)
    print("=== 任务分解 ===")
    print(plan)
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
        model="glm-4.6",
        temperature=1.0,
        messages=[
            {"role": "system", "content": "请用自然语言总结以下工具调用过程和结果。"},
            {"role": "user", "content": summary}
        ]
    )
    print("=== 汇总结果 ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
