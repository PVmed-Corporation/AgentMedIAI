from pathlib import Path
from typing import Dict, Any
import json
from zai import ZhipuAiClient
from generate_tool_info import build_tools_prompt
from retriever import ToolRetriever
import os

SYSTEM_PROMPT = """
You are an AI agent that translates user queries into executable tool calls.
You will be given:
1. A user query describing a task.
2. A list of available tools with their descriptions, demo commands, and schemas.

### Your task:
Generate a JSON output in the following schema:
{
  "tool_name": "Name of the selected tool.",
  "explanation": "A short human-readable explanation of what the tool call will do.",
  "command": "The exact command string to run, either as a Python call or as a Linux CLI command."
}

### Rules for deciding "command" format:
1. If the tool provides demo commands, docker commands, or examples with flags like `-i`, `-o`, `--mode`, then assume the tool is a **CLI tool**.
   - In this case, output the `command` field as a **Linux CLI string** (e.g., `cxas_feat_extract -i input.png -o output_dir -f CTR`).
   - Follow the exact CLI syntax shown in the tool’s demo commands.
   - Do not invent parameters.
   - If an argument value contains spaces (e.g., Cardio-Thoracic Ratio), wrap it in single quotes `'...'`.
   - Never wrap arguments in double quotes (`"..."`) unless explicitly required by the tool — use single quotes instead.
   - Always use Linux-style paths with forward slashes (`/`), not Windows-style backslashes (`\\`).
   - Always choose argument values from the tool’s allowed options.
2. If the tool does not provide CLI examples but only a Python API or function signature, assume it is a **Python tool**.
   - In this case, output the `command` field as a **Python function call string** (e.g., `CTRCalculator(input="...", output="...")`).
3. Always respect the input/output schema provided with the tool.
   - Fill in required parameters (like input/output paths).
   - Use default values if specified.
   - Only include optional parameters if relevant.
4. Only output valid JSON that can be parsed with `json.loads()`.
   - Do not include any text, comments, or explanations outside of the JSON object.
"""

class ToolCallAgent:
    """
    - 检索工具
    - 构造工具提示词
    - 调用大模型生成工具调用 JSON
    """

    def run(self, query: str, tools_path: str, input_path: Path, result_path: Path) -> Dict[str, Any]:
        # 1. 检索可用工具
        retriever = ToolRetriever()
        selected = retriever.prompt_based_retrieval(query, tools_path)

        # 2. 构造工具提示词
        tools_prompt = build_tools_prompt(selected, "tool")

        # 3. 调用大模型
        client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
        response = client.chat.completions.create(
            model="glm-4.5",
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User query: {query}\n\n"
                        f"Available tools:\n{json.dumps(tools_prompt, indent=2)}\n\n"
                        f"Input path: {input_path}\n"
                        f"Result path: {result_path}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        # 4. 解析 JSON
        result = json.loads(response.choices[0].message.content)
        return result
