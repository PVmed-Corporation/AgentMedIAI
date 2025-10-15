import contextlib
import re
import os
from zai import ZhipuAiClient
import json



class ToolRetriever:
    """Retrieve tools from the tool registry."""
    def __init__(self):
        pass

    def prompt_based_retrieval(self, query: str, tools_or_path) -> dict:
        """使用提示词检索最相关工具。
        Args:
            query: 用户查询
            tools_or_path: - str: JSON/JSONL 文件路径
        Returns:
            dict: {"tools": [被选中的工具列表]}
        """
        # 1) 标准化为列表，确保与展示顺序一致，用于索引映射与回传
        if isinstance(tools_or_path, str) and os.path.isfile(tools_or_path):
            tools_list = self._load_tools_from_json_file(tools_or_path) or []
        elif isinstance(tools_or_path, dict):
            tools_list = tools_or_path.get("tools", []) or []
        elif isinstance(tools_or_path, list):
            tools_list = tools_or_path
        else:
            tools_list = []
        #如果工具少于10就不适用LLM检索
        if len(tools_list) < 10:
            print(f"[INFO] Tool count ({len(tools_list)}) < 10 → skipping LLM retrieval.")
            return {"tools": tools_list}
        # 2) 构造提示：展示层直接使用 tools_or_path，由 _format_tools_for_prompt 负责格式化
        prompt = f"""
        You are an expert medical AI assistant. Your task is to select the most relevant tools to help answer a clinician's query.

        USER QUERY: {query}

        Below are the available tools. Each tool includes its MODALITY, NAME, and DESCRIPTION.
        Your goal is to carefully analyze the user's query, understand the imaging modality and the medical task, and then select all tools that are directly or indirectly useful for solving the query.

        Be generous in your selection: include tools that might be useful, even if they are not explicitly mentioned in the query. It's better to include slightly more tools than to miss potentially relevant ones.

        AVAILABLE TOOLS:
        {self._format_tools_for_prompt(tools_or_path)}

        Please respond with ONLY the indices of the relevant tools in the following format:

        TOOLS: [list of indices]

        For example:
        TOOLS: [0, 2, 4, 7]

        If no tools are relevant, respond with: TOOLS: []

        IMPORTANT GUIDELINES:
            1. If the user query explicitly specifies a modality (e.g., CT, Neuroimaging(MR), X-ray, Ultrasound, Fundus, OCT, Dermoscopy,Dermatology Clinical (Derm-Clinic) , Endoscopic Images(Endo), Neuroimaging, Text ), ONLY select tools of that modality. Ignore tools from other modalities.
            2. If the modality is not specified, include all potentially relevant tools regardless of modality.
            3. Prefer tools that can directly measure or analyze the requested anatomical structure, pathology, or clinical parameter.  
            4. If the query involves measurement (e.g., organ volume, lesion size), include segmentation or quantification tools.  
            5. If the query involves detection, diagnosis, or classification, include diagnostic or screening tools.  
            6. If the query is general or ambiguous, include all potentially relevant tools for the given modality.  
            7. When in doubt about a tool, include it rather than exclude it.  
            8. Do NOT output any explanation—only the tool indices in the specified format.  
        """
        client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))
        response = client.chat.completions.create(
            model="glm-4.5",
            # thinking={ "enabled": True},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        response_content = response.choices[0].message.content
        # 3) 解析索引并基于 tools_list 回传真实对象，保证与展示一致
        selected_indices = self._parse_llm_response(response_content)
        print(selected_indices)#{'tools': [0, 1, 2, 3, 4, 5]}
        selected_tools = {
            "tools": [
                tools_list[i] for i in selected_indices.get("tools", []) if 0 <= i < len(tools_list)
            ],
        }
        print(f"selected tools: {selected_tools}")
        return selected_tools

    def _format_tools_for_prompt(self, tools_or_path) -> str:
        """支持两种输入:
        1) 字符串路径 -> 从 JSON/JSONL 文件读取工具列表
        2) list -> 直接使用传入的工具列表
        """
        if isinstance(tools_or_path, str) and os.path.isfile(tools_or_path):
            tools = self._load_tools_from_json_file(tools_or_path)
        else:
            tools = tools_or_path or []

        formatted = []
        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                name = tool.get("tool_name") or tool.get("name") or f"Resource {i}"
                desc = tool.get("tool_description") or tool.get("description") or ""
                modality = tool.get("modality")
                if modality:
                    formatted.append(f"{i}. [{modality}] {name}: {desc}")
                else:
                    formatted.append(f"{i}. {name}: {desc}")
            elif isinstance(tool, str):
                formatted.append(f"{i}. {tool}")
            else:
                name = getattr(tool, "name", str(tool))
                desc = getattr(tool, "description", "")
                modality = getattr(tool, "modality", None)
                if modality:
                    formatted.append(f"{i}. [{modality}] {name}: {desc}")
                else:
                    formatted.append(f"{i}. {name}: {desc}")

        return "\n".join(formatted) if formatted else "None available"

    def _load_tools_from_json_file(self, path: str) -> list:
        """从 JSON/JSONL 文件加载工具:
        - .jsonl: 一行一个 JSON 对象
        - .json: 数组或单对象；若文件缺少外层 [] 但用逗号分隔对象，也会尝试自动包裹
        """
        try:
            if path.lower().endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]

            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # 尝试标准 JSON
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return [data]
                if isinstance(data, list):
                    return data
                return []
            except json.JSONDecodeError:
                try:
                    data = json.loads(f"[{text}]")
                    if isinstance(data, list):
                        return data
                except json.JSONDecodeError:
                    return []
        except OSError:
            return []

    def _parse_llm_response(self, response: str) -> dict:
        """Parse the LLM response to extract the selected indices."""
        selected_tools = {"tools": []}

        # Extract indices for each category
        tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if tools_match and tools_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_tools["tools"] = [int(idx.strip()) for idx in tools_match.group(1).split(",") if idx.strip()]
        return selected_tools
