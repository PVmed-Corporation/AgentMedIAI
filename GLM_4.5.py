from zai import ZhipuAiClient
import os

# 初始化客户端
client = ZhipuAiClient(api_key=os.environ.get("API_KEY"))

# 创建聊天完成请求
response = client.chat.completions.create(
    model="glm-4.5",
    messages=[
        {
            "role": "system",
            "content": "你是一个有用的AI助手。"
        },
        {
            "role": "user",
            "content": "你好，请介绍一下自己。"
        }
    ],
    temperature=0.6
)

# 获取回复
print(response.choices[0].message.content)

