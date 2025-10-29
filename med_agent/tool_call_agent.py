from pathlib import Path
from typing import Dict, Any
import json

from triton.language.semantic import truediv
from zai import ZhipuAiClient

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
  "command_type": "One of 'cli', 'python', or 'docker'.",
  "command": "The exact command string to run, either as a Python call, a Linux CLI command, or a docker command."
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
   - If the tool’s input schema does not include an output path parameter but the user specifies one, ignore the user-provided output path and proceed according to the correct format required by the tool invocation.

4. The generated command serves only as a sub-command string and is not executed directly. At runtime,
  -This sub-command will be wrapped by the external execution logic into a complete runnable command.
    CLI commands are wrapped as: conda run -n {env_name} bash -c "{command}"
    Python commands are executed via _run_python(), which automatically writes the code into a temporary .py file and runs it inside the conda environment.
    Therefore, you are only responsible for generating the core executable content ({command}) ,
    the plain Python code (for Python type) or the raw CLI command (for CLI type), without conda run, quotes, or shell wrappers.
5. Only output valid JSON that can be parsed with `json.loads()`.
   - Do not include any text, comments, or explanations outside of the JSON object.
6. tools were deployed with GPU support
"""
DEBUG_SYSTEM_PROMPT = """
You are now in DEBUG MODE.

The last tool call failed. Analyze the cause and generate a corrected tool call JSON.

You will be given:
1. The previous failed command.
2. The observed error message (if any).
3. The available tool list remains the same.

### Your task:
Return the corrected JSON in the same schema.
if tool provide lost of model ,you need to chose the best suitable model 
If the error message provides usage information or indicates the expected format, you MUST modify the command strictly following that guidance. 
Use the error details to identify and correct missing parameters, incorrect field names, or invalid syntax.
if last tool is no suitable for this task, you can change to use another tool from the available tool list.

Example:
{
  "tool_name": "Name of the selected tool.",
  "explanation": "bug reason and how to fix it",
  "command_type": "One of 'cli', 'python', or 'docker'.",
  "command": "The exact command string to run, either as a Python call, a Linux CLI command, or a docker command."
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
   - If the tool’s input schema does not include an output path parameter but the user specifies one, ignore the user-provided output path and proceed according to the correct format required by the tool invocation.

4. The generated command serves only as a sub-command string and is not executed directly. At runtime,
   -This sub-command will be wrapped by the external execution logic into a complete runnable command.
    CLI commands are wrapped as: conda run -n {env_name} bash -c "{command}"
    Python commands are executed via _run_python(), which automatically writes the code into a temporary .py file and runs it inside the conda environment.
    Therefore, you are only responsible for generating the core executable content ({command}) ,
    the plain Python code (for Python type) or the raw CLI command (for CLI type), without conda run, quotes, or shell wrappers.
5. Only output valid JSON that can be parsed with `json.loads()`.
   - Do not include any text, comments, or explanations outside of the JSON object.
6. tools were deployed with GPU support
Make only minimal, necessary corrections.
"""

class ToolCallAgent:
    """
    - 检索工具
    - 构造工具提示词
    - 调用大模型生成工具调用 JSON
    """
    def __init__(self,memory):
        self.memory = memory
    def run(
        self,
        query: str,
        input_path: Path,
        output_path: Path,
        tools_prompt: str,
        bug: bool = False,
        error_msg: str = "",
    ) -> Dict[str, Any]:
        client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
        if not bug:
            relevant_memories = self.memory.search(query=query, user_id="default_user", limit=3)
            memories_text = "\n".join(f"- {m['memory']}" for m in relevant_memories["results"])

            # Step 2: 构造带记忆的系统提示词
            system_prompt_with_memory = SYSTEM_PROMPT + f"\n\n---\nRelevant past context:\n{memories_text}\n---\n"

            response = client.chat.completions.create(
                model="glm-4.6",
                temperature=1,
                thinking={"type": "enabled"},
                messages=[
                    {"role": "system", "content": system_prompt_with_memory},
                    {
                        "role": "user",
                        "content": (
                            f"User query: {query}\n\n"
                            f"Available tools:\n{json.dumps(tools_prompt, indent=2)}\n\n"
                            f"Input path: {input_path}\n"
                            f"Result path: {output_path}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )

            # 解析 JSON
            result = json.loads(response.choices[0].message.content)
            return result

        else:
            response = client.chat.completions.create(
                model="glm-4.6",
                temperature=1,
                messages=[
                    {"role": "system", "content": DEBUG_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"User query: {query}\n\n"
                            f"Available tools:\n{json.dumps(tools_prompt, indent=2)}\n\n"
                            f"error message: {error_msg}\n\n"
                            f"Input path: {input_path}\n"
                            f"Result path: {output_path}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )

            # 4. 解析 JSON
            result = json.loads(response.choices[0].message.content)
            return result

