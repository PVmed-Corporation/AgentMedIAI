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

from openai import OpenAI
from mem0 import Memory


def create_memory():
    config = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 1.0
            }
        }

    }

    return Memory.from_config(config)

def run_subtask(task, tools_path, memory,max_retries=4):
    memory = memory
    tool_call_agent = ToolCallAgent(memory=memory)
    tool_invoker = ToolInvoker()
    sub_query = task["sub_query"]
    # 有些任务不需要输入输出路径
    input_path = Path(task["input_path"]) if task.get("input_path") else None
    output_path = Path(task["output_path"]) if task.get("output_path") else None
    attempt = 0
    result = None
    tool_call_json = None
    error_msg = ""
    """
    tool_call_json schema:
    {
      "tool_name
      "explanation"
      "command_type"
      "command"
    }

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

    retriever = ToolRetriever()
    selected = retriever.prompt_based_retrieval(sub_query, tools_path)
    tools_prompt = build_tools_prompt(selected, "tool")

    while attempt < max_retries:
        tool_call_json = tool_call_agent.run(sub_query, input_path, output_path, tools_prompt, bug=(attempt > 0),
                                             error_msg=error_msg)
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

    # 构造总结文本
    summary = "工具调用结果如下：\n"
    summary += (
        f"子任务: {sub_query}\n"
        f"工具: {tool_call_json.get('tool_name') if tool_call_json else 'N/A'}\n"
        f"命令: {tool_call_json.get('command') if tool_call_json else 'N/A'}\n"
        f"结果: {json.dumps(result, ensure_ascii=False)}\n"
        f"尝试次数: {attempt + 1}\n"
        "----------------------\n"
    )

    # 使用 Zhipu GLM 生成总结
    client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
    messages = [
        {"role": "system", "content": "请用自然语言总结以下调用过程,包括调用命令，调用失败的的报错及其原因，调用成功的结果总结。"},
        {"role": "user", "content": summary}
    ]
    response = client.chat.completions.create(
        model="glm-4.6",
        temperature=1.0,
        messages=messages
    )
    print("=== 汇总结果 ===")
    print(response.choices[0].message.content)
    natural_summary = response.choices[0].message.content
    memory.add(natural_summary, user_id="default_user",metadata={"category": "tool_invoke_summary"})
    messages.append({"role": "assistant", "content": natural_summary})
    # 是否有必要吧summary一起加进toolcallagent的记忆中
    memory.add(messages, user_id="default_user", metadata={"category": "tool_invoke_summary,tool_invoke_result"})

    return {
        "tool_name": tool_call_json.get("tool_name") if tool_call_json else None,
        "command": tool_call_json.get("command") if tool_call_json else None,
        "result": result,
        "summary": natural_summary,
        "sub_query": sub_query,
        "input_path": str(input_path) if input_path else None,
        "output_path": str(output_path) if output_path else None,
        "attempts": attempt + 1
    }


def main():
    tools_path = "tools.json"
    input_path = "/data/result/zhangjie/image/ETT.jpg"
    result_path = "/data/result/yilinyou/CXAS_result/"
    os.environ["OPENAI_API_KEY"] =
    os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
    os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
    os.environ["DEEPSEEK_API_KEY"] =
    memory = create_memory()
    planning_agent = PlanningAgent(memory=memory)
    print("chat with ai(type 'exit' to quit)")
    while True:
        query = input("You: ").strip()
        # query = f"{query}\n input image path: {input_path}\noutput image path: {result_path}"
        # 初始化变量
        current_result = None
        cycle_count = 0

        while cycle_count < 3:
            cycle_count += 1
            print(f"\n===== 第 {cycle_count} 轮规划 =====")

            # 调用规划器（携带上轮结果）
            print(f"第{cycle_count}规划的query{query}")
            plan = planning_agent.run(query)

            # 如果是聊天型输出 -> 结束循环
            if plan.get("type") == "chat":
                print("\n=== 最终回答 ===")
                print(plan.get("respond", "（无内容）"))
                break

            # 如果是任务型输出 -> 执行工具调用
            elif plan.get("type") == "task":
                print("\n=== 执行任务 ===")
                task = plan
                subtask_result = run_subtask(task, tools_path,memory)

                # 过滤掉 result 部分，仅保留摘要和元信息
                current_result = {
                    k: v for k, v in subtask_result.items()
                    if k != "result"
                }

                print("\n=== 工具调用完成，反馈给规划器 ===")
                print(json.dumps(current_result, indent=4, ensure_ascii=False))
                query = (
                    "上一步工具执行的结果如下：\n"
                    f"{json.dumps(current_result, indent=2, ensure_ascii=False)}\n\n"
                    "请基于以上信息继续规划下一步操作或输出最终结论。"
                )


            # 如果规划器返回空或错误
            else:
                print(" 未识别的规划类型或空任务，结束。")
                break

        else:
            print("达到最大循环次数，任务终止。")


if __name__ == "__main__":
    main()
