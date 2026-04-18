import ast, csv, json, operator
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from langchain_core.tools import tool

from .load import openweather_api_key, model


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


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间。参数 timezone 形如 'Asia/Shanghai' / 'America/New_York' / 'UTC'，默认上海。"""

    try:
        now = datetime.now(ZoneInfo(timezone))
        return f"{timezone} 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as exc:
        return f"时间查询失败：{exc}"


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
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。支持 + - * / // % ** 和括号，例如 '(3+4)*2'。"""

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"计算失败：{exc}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """货币汇率换算。参数：amount 金额；from_currency 源币种代码(如 USD)；to_currency 目标币种代码(如 CNY)。"""

    base = from_currency.upper()
    target = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{base}"

    try:
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("result") != "success":
            return f"汇率查询失败：{data.get('error-type', 'unknown error')}"

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
    url = f"https://ipapi.co/{target + '/' if target else ''}json/"

    try:
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
def read_file(path: str) -> str:
    """
    读取不同类型的文件并进行智能解析。
    参数: path (str): 文件路径
    返回: str: AI 对文件内容的分析结果
    """
    path = Path(path)
    suf = path.suffix
    if suf in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8") as file:
             data = file.read()
    elif suf == ".json":
        with open(path, "r", encoding="utf-8") as file:
            obj = json.load(file)
            data = json.dumps(obj, ensure_ascii=False, indent=2)
    elif suf == ".csv":
        with open(path, "r", encoding="utf-8") as file:
            rows = list(csv.reader(file))
            data = json.dumps(rows, ensure_ascii=False)
    elif suf == ".xlsx":
            df = pd.read_excel(path)
            data = df.to_csv(index=False)
    else:
          return f"不支持的文件类型: {suf}"
    resp = model.invoke(f"分析这个文件，判断类型并解释内容，"
                 f"包括结构、用途和关键信息。"
                 f"如果是代码就解释逻辑，如果是数据文件就说明字段，如果是二进制文件就说明用途。"
                 f"以下是文件:\n{data}")
    return resp.content

tools = [check_weather, get_current_time, calculate, convert_currency, lookup_ip, read_file]
tools_by_name = {tool.name: tool for tool in tools}
