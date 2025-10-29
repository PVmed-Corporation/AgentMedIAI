from openai import OpenAI
import os
from mem0 import Memory
os.environ["OPENAI_API_KEY"] = "sk-M7X461785fa09b78f3f978edecede2d7895769f1ea7sZDdG"
os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"
os.environ["OPENAI_API_BASE"] = "https://api.gptsapi.net/v1"
os.environ["DEEPSEEK_API_KEY"] = "sk-f1470a8f46ad4372a536f75393d38c6f"

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
        },
        "embedding": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": "https://api.gptsapi.net/v1"
            }
        }
    }

    return Memory.from_config(config)