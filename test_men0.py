import os
from mem0 import Memory

# DeepSeek 用于 LLM
os.environ["DEEPSEEK_API_KEY"] = "sk-f1470a8f46ad4372a536f75393d38c6f"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 不再设置 OPENAI_BASE_URL（删除这行）
os.environ["OPENAI_BASE_URL"] = "https://api.gptsapi.net/v1"

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
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
}

m = Memory.from_config(config)

messages = [
    {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
    {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
    {"role": "user", "content": "I’m not a big fan of thriller movies but I love sci-fi movies."},
    {"role": "assistant", "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future."}
]

m.add(messages, user_id="alice", metadata={"category": "movies"})
print(" Memory added successfully")
