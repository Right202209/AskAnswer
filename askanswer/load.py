import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

load_dotenv(override=True)

model = init_chat_model(
    "gpt-5.4",
    temperature=0,
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
