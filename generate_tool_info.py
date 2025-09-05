
from pathlib import Path
from typing import List, Dict, Any
import json


def extract_tool_names(selected: Any) -> List[str]:
    """
    将 selected 统一转换为工具名列表。
    支持:
      - {"tools": [ { "tool_name": "..."} | "name", ... ]}
      - [ { "tool_name": "..."} | "name", ... ]
    """
    tools = selected
    if isinstance(selected, dict):
        tools = selected.get("tools", [])

    names: List[str] = []
    seen = set()

    if isinstance(tools, list):
        for item in tools:
            name = None
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = item.get("tool_name") or item.get("name") or item.get("id")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def build_tools_prompt(selected: Any, tools_dir: str | Path) -> Dict[str, Any]:
    """
    给定工具名列表与目录, 读取同名 .json 并拼成 prompt 的 function 段。
    若无可用工具则返回 {}。
    """
    selected_tools = extract_tool_names(selected)
    tools_dir = Path(tools_dir)
    if not selected_tools:
        return {}

    functions: List[Dict[str, Any]] = []
    for name in selected_tools:
        file_path = tools_dir / f"{name}.json"
        if not file_path.is_file():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as f:
                tool_spec = json.load(f)
            functions.append(tool_spec)
        except Exception:
            continue

    return {"function": functions} if functions else {}

