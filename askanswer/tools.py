import json
from urllib.parse import urlencode
from urllib.request import urlopen

from langchain_core.tools import tool

from .load import openweather_api_key


@tool
def check_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为城市名，例如 Beijing。"""

    if not openweather_api_key:
        return "未配置 OPENWEATHER_API_KEY"

    query = urlencode(
        {
            "q": city,
            "appid": openweather_api_key,
            "units": "metric",
            "lang": "zh_cn",
        }
    )
    url = f"https://api.openweathermap.org/data/2.5/weather?{query}"

    try:
        with urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        if str(data.get("cod")) != "200":
            return f"天气查询失败：{data.get('message', 'unknown error')}"

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        city_name = data["name"]

        return (
            f"{city_name} 当前天气：{weather}，"
            f"气温 {temp}°C，体感 {feels_like}°C，湿度 {humidity}%"
        )
    except Exception as exc:
        return f"天气查询失败：{exc}"


tools = [check_weather]
tools_by_name = {tool.name: tool for tool in tools}
