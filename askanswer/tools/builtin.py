"""Read-only built-in tools: weather, time, calculator, currency, IP lookup, cwd, web search.

None of these tools has side effects on the host, so they are exposed to every
intent tag.  External clients and API keys are resolved lazily at call time –
importing this module never reads credentials or opens connections.
"""

from __future__ import annotations

import ast
import json
import operator
import os
from datetime import datetime
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

_tavily_client = None
_tavily_lock = Lock()


def get_tavily_client():
    """Return the process-wide Tavily client, built on first use from ``TAVILY_API_KEY``."""
    global _tavily_client
    with _tavily_lock:
        if _tavily_client is None:
            from tavily import TavilyClient

            from ..config.settings import ensure_bootstrapped

            ensure_bootstrapped()
            _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        return _tavily_client


def reset_tavily_client() -> None:
    """Drop the cached Tavily client (key rotation / tests)."""
    global _tavily_client
    with _tavily_lock:
        _tavily_client = None


def openweather_api_key() -> str | None:
    """OpenWeather API key, read at call time (``OPENWEATHER_API_KEY``)."""
    from ..config.settings import ensure_bootstrapped

    ensure_bootstrapped()
    return os.getenv("OPENWEATHER_API_KEY") or None


@tool
def check_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为城市名，例如 Beijing。"""

    # API Key 缺失时给出明确提示，而不是抛异常 —— 更友好且不影响图流程。
    # 调用期读取（而非导入期固化），settings.json 惰性 bootstrap 后写入的 key 也能生效。
    api_key = openweather_api_key()
    if not api_key:
        return "未配置 OPENWEATHER_API_KEY"

    # 拼接 OpenWeather One Call 接口的查询串：公制单位 + 中文描述
    query = urlencode(
        {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "zh_cn",
        }
    )
    url = f"https://api.openweathermap.org/data/2.5/weather?{query}"

    try:
        with urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

        # 接口返回 cod 字段表示状态；非 200 视为查询失败
        if str(data.get("cod")) != "200":
            return f"天气查询失败：{data.get('message', 'unknown error')}"

        # 解析关键字段并组装为人类可读的一句话
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
        # 网络/解析异常都包成字符串返回，避免 ToolNode 因异常而报错
        return f"天气查询失败：{exc}"

@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间。参数 timezone 形如 'Asia/Shanghai' / 'America/New_York' / 'UTC'，默认上海。"""

    try:
        # ZoneInfo 是 Python 3.9+ 的官方时区库，无需额外依赖
        now = datetime.now(ZoneInfo(timezone))
        return f"{timezone} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as exc:
        # 时区拼写错误等会走到这里
        return f"时间查询失败：{exc}"


# 计算器允许的 AST 操作符白名单：避免 eval 任意代码注入
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node: ast.AST) -> float:
    """递归地对 AST 节点求值，仅允许白名单操作符与数字字面量。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        # 二元运算：左右子节点先递归求值再用对应 operator 函数
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    # 任何不在白名单内的语法都直接拒绝
    raise ValueError("不支持的表达式")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。支持 + - * / // % ** 和括号，例如 '(3+4)*2'。"""

    try:
        # 用 ast.parse 而非 eval：可以严格控制可执行的语法
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """货币汇率换算。参数：amount 金额；from_currency 源币种代码(如 USD)；to_currency 目标币种代码(如 CNY)。"""

    # 统一转大写，open.er-api 接口对大小写敏感
    base = from_currency.upper()
    target = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{base}"

    try:
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("result") != "success":
            return f"汇率查询失败：{data.get('error-type', 'unknown error')}"

        # rates 是 {"USD": 1.0, "CNY": 7.x, ...} 这样的字典
        rate = data.get("rates", {}).get(target)
        if rate is None:
            return f"未找到 {base} → {target} 的汇率"

        return f"{amount} {base} ≈ {amount * rate:.4f} {target}（汇率 {rate}）"
    except Exception as exc:
        return f"汇率查询失败：{exc}"


@tool
def lookup_ip(ip: str = "") -> str:
    """查询 IP 地址的地理位置与运营商信息。参数 ip 留空则查询当前出口 IP。"""

    target = ip.strip()
    # ipapi 的接口形式：/{ip}/json/，留空则查询调用端自身公网 IP
    url = f"https://ipapi.co/{target + '/' if target else ''}json/"

    try:
        # 显式带 UA：部分免费接口会拒绝缺省 UA
        request = Request(url, headers={"User-Agent": "AskAnswer/0.1"})
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("error"):
            return f"IP 查询失败：{data.get('reason', 'unknown error')}"

        return (
            f"IP {data.get('ip')}：{data.get('country_name')} "
            f"{data.get('region')} {data.get('city')}，"
            f"运营商 {data.get('org')}，时区 {data.get('timezone')}"
        )
    except Exception as exc:
        return f"IP 查询失败：{exc}"


@tool
def tavily_search(query: str) -> str:
    """联网搜索实时或最新信息。参数 query 为搜索关键词，返回 Top 5 网页摘要与 Tavily 综合回答。"""

    try:
        # search_depth=basic 速度快；include_answer 让 Tavily 顺带给一个综合摘要
        response = get_tavily_client().search(
            query=query, search_depth="basic", max_results=5, include_answer=True
        )
    except Exception as exc:
        return f"搜索失败：{exc}"

    results = response.get("results", [])
    tavily_answer = (response.get("answer") or "").strip()

    # 拼装为 Markdown 形式，便于 LLM 直接引用并呈现给用户
    out = f"查询关键词：{query}\n\n"
    if tavily_answer:
        out += f"Tavily 摘要：{tavily_answer}\n\n"
    out += "搜索结果（Top 5）：\n\n"

    if not results:
        out += "未找到任何搜索结果。\n"
        return out.strip()

    for i, result in enumerate(results, 1):
        title = result.get("title", "无标题")
        url = result.get("url", "#")
        content = result.get("content", "无内容摘要")
        score = result.get("score", 0.0)
        # 内容过长时截断，避免 prompt 爆炸
        out += (
            f"{i}. **{title}** (相关度: {score:.3f})\n"
            f"   链接: {url}\n"
            f"   {content[:280]}{'...' if len(content) > 280 else ''}\n\n"
        )
    return out.strip()


@tool
def pwd()->str:
    """
    获取当前工作目录的路径
    :param pwd:
    :return: 当前工作目录的路径
    """
    current_directory = os.getcwd()
    return current_directory
