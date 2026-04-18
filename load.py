from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)  # 加载 .env 文件

# 验证是否加载成功
# print("API Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

model = init_chat_model(
    "gpt-5.4",
    temperature=0
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
