import subprocess
import sys
import traceback
from typing import Dict, Any


class ToolInvoker:
    """
    负责执行 ToolCallAgent 生成的命令。
    """

    def invoke(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        env_name= tool_call["tool_name"]# 获取工具对应的 conda 环境名称,conda环境名称与工具名称相同，当时配置的时候有出现几个特例
        if env_name == "ChestXRayAnatomySegmentation":
            env_name = "CXAS"
        command = tool_call["command"]
        command_type = tool_call.get("command_type", "cli").lower()

        if command_type == "cli" or command_type == "docker":
            return self._run_cli(command,env_name)
        elif command_type == "python":
            return self._run_python(command, env_name)
        else:
            return {
                "status": "error",
                "stderr": f"Unknown command_type: {command_type}"
            }

    def _run_cli(self, command: str,env_name:str,timeout: int=500) -> Dict[str, Any]:
        if env_name:
            command=f"conda run -n {env_name} {command}"
        else:
            command=f"conda run -n base {command}"
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
        except subprocess.TimeoutExpired as e:
            return {
                "status": "error",
                "stderr": f"Command timed out after {timeout} seconds",
                "stdout": e.stdout or "",
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "stdout": e.stdout.strip() if e.stdout else "",
                "stderr": e.stderr.strip() if e.stderr else "",
                "returncode": e.returncode
            }

    def _run_python(self, code: str, env_name: str = None, timeout: int = 500) -> Dict[str, Any]:
        if env_name:
            command = f'conda run -n {env_name} python -c "{code}"'
        else:
            command = f'conda run -n base python -c "{code}"'
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "status": "success",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except subprocess.TimeoutExpired as e:
            return {
                "status": "error",
                "stderr": f"Python code timed out after {timeout} seconds",
                "stdout": e.stdout or "",
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "stdout": e.stdout.strip() if e.stdout else "",
                "stderr": e.stderr.strip() if e.stderr else "",
                "returncode": e.returncode
            }