import subprocess
import sys
import traceback
from typing import Dict, Any
import time

class ToolInvoker:
    """
    负责执行 ToolCallAgent 生成的命令。
    """

    def invoke(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        env_name = tool_call["tool_name"]
        if env_name == "ChestXRayAnatomySegmentation":
            env_name = "CXAS"
        command = tool_call["command"]
        command_type = tool_call.get("command_type", "cli").lower()

        if command_type == "cli" or command_type == "docker":
            return self._run_cli(command, env_name)
        elif command_type == "python":
            return self._run_python(command, env_name)
        else:
            return {"status": "error", "stderr": f"Unknown command_type: {command_type}"}

    def _run_cli(self, command: str, env_name: str, timeout: int = 500) -> Dict[str, Any]:
        if env_name:
            command = f'conda run -n {env_name} bash -c "{command}"'
        else:
            command = f'conda run -n base bash -c "{command}"'

        print(f"\n[INFO] Executing command: {command}")
        sys.stdout.flush()

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # 行缓冲
        )

        output_lines = []
        start_time = time.time()

        try:
            for line in process.stdout:
                print(line, end="")  # 实时打印
                sys.stdout.flush()
                output_lines.append(line)

            process.wait(timeout=timeout)

            elapsed = time.time() - start_time
            status = "success" if process.returncode == 0 else "error"
            return {
                "status": status,
                "stdout": "".join(output_lines).strip(),
                "stderr": "",
                "returncode": process.returncode,
                "runtime_sec": round(elapsed, 2)
            }

        except subprocess.TimeoutExpired:
            process.kill()
            return {"status": "error", "stderr": f"Timeout after {timeout}s"}

    def _run_python(self, command: str, env_name: str = None, timeout: int = 500) -> Dict[str, Any]:
        if env_name:
            command = f'conda run -n {env_name} python -c "{command}"'
        else:
            command = f'conda run -n base python -c "{command}"'

        print(f"\n[INFO] Running Python command in {env_name}:")
        print(command)
        sys.stdout.flush()

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        start_time = time.time()

        try:
            for line in process.stdout:
                print(line, end="")
                sys.stdout.flush()
                output_lines.append(line)

            process.wait(timeout=timeout)

            elapsed = time.time() - start_time
            status = "success" if process.returncode == 0 else "error"
            return {
                "status": status,
                "stdout": "".join(output_lines).strip(),
                "stderr": "",
                "returncode": process.returncode,
                "runtime_sec": round(elapsed, 2)
            }

        except subprocess.TimeoutExpired:
            process.kill()
            return {"status": "error", "stderr": f"Timeout after {timeout}s"}