import subprocess
import sys
import traceback
from typing import Dict, Any


class ToolInvoker:
    """
    负责执行 ToolCallAgent 生成的命令。
    """

    def invoke(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        command = tool_call["command"]

        # 判断调用类型
        if command.strip().startswith("docker ") or command.strip().split()[0].isalpha():
            # CLI 或 docker
            return self._run_cli(command)
        else:
            # Python 调用（简单判断: 包含() 且不是 shell 命令）
            return self._run_python(command)

    def _run_cli(self, command: str) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            return {
                "status": "success",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "stdout": e.stdout.strip() if e.stdout else "",
                "stderr": e.stderr.strip() if e.stderr else "",
                "returncode": e.returncode
            }

    def _run_python(self, code: str) -> Dict[str, Any]:
        try:
            local_vars = {}
            exec(code, {}, local_vars)  # ⚠️ 注意：这里执行用户生成的代码有安全风险
            return {
                "status": "success",
                "locals": local_vars
            }
        except Exception:
            return {
                "status": "error",
                "stderr": traceback.format_exc()
            }
