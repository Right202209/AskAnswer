import ast, csv, json, operator, re
import os
import platform
import shlex
import subprocess
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from markitdown import MarkItDown
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
    data = markitdown(path)
    resp = model.invoke(f"分析这个文件,解释内容，"
                 f"包括结构、用途和关键信息。"
                 f"如果是代码就解释逻辑，如果是数据文件就说明字段，如果是二进制文件就说明用途。"
                 f"以下是文件:\n{data}")
    return resp.content

def markitdown(file_path: str) -> str:
    md = MarkItDown(enable_plugins=True)
    result = md.convert(file_path)
    return result.text_content


@tool
def pwd()->str:
    """
    获取当前工作目录的路径
    :param pwd:
    :return: 当前工作目录的路径
    """
    current_directory = os.getcwd()
    return current_directory


_DANGEROUS_PATTERNS = [
    (r"\brm\b", "rm 删除命令"),
    (r"\brmdir\b", "rmdir 目录删除"),
    (r"\bshutdown\b", "shutdown 关机"),
    (r"\breboot\b", "reboot 重启"),
    (r"\bhalt\b", "halt 停机"),
    (r"\bpoweroff\b", "poweroff 关机"),
    (r"\bmkfs\w*\b", "mkfs 格式化"),
    (r"(?:^|\s)dd\s+if=", "dd 底层磁盘写入"),
    (r":\s*\(\)\s*\{", "fork bomb"),
    (r"\bsudo\b", "sudo 提权"),
    (r"(?<!>)>(?![>&])", "重定向覆盖 `>`（会覆盖文件）"),
    (r"\bkill\s+-9\b", "kill -9 强制终止"),
    (r"\bchmod\s+-R\b", "chmod -R 递归改权限"),
    (r"\bchown\s+-R\b", "chown -R 递归改属主"),
    (r">\s*/dev/sd", "写入磁盘设备"),
    (r"\bmv\s+\S+\s+/\b", "mv 到根目录"),
]


def _check_dangerous(command: str) -> str | None:
    for pattern, desc in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return desc
    return None


def _clean_command(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s.lstrip("$").strip()


def gen_shell_command_spec(instruction: str) -> tuple[str, str]:
    current_os = platform.system() + " " + platform.release() + " " + platform.machine()
    prompt = (
        f"根据以下用户指令生成在 {current_os} 环境下可直接执行的单条 shell 命令。\n"
        f"严格按以下两行格式输出，不要 markdown、不要多余内容：\n"
        f"命令：<shell command>\n"
        f"说明：<一句话解释它做什么>\n\n"
        f"用户指令：{instruction}"
    )
    resp = model.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    command = ""
    explanation = ""
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("命令：", "命令:"):
            if stripped.startswith(prefix):
                command = stripped[len(prefix):].strip()
                break
        for prefix in ("说明：", "说明:"):
            if stripped.startswith(prefix):
                explanation = stripped[len(prefix):].strip()
                break
    command = _clean_command(command or text)
    return command, explanation


def execute_shell_command(command: str, shell: bool = False) -> str:
    popen_args: str | list[str]
    if shell:
        popen_args = command
    else:
        try:
            popen_args = shlex.split(command)
        except ValueError as exc:
            return f"命令解析失败：{exc}\n原始命令：{command}"

    try:
        process = subprocess.Popen(
            popen_args,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return f"命令执行超时（>30s）：{command}"
    except FileNotFoundError as exc:
        return f"命令未找到：{exc}"
    except Exception as exc:
        return f"命令执行失败：{exc}"

    ok = process.returncode == 0
    parts = [
        f"命令：{command}",
        f"返回码：{process.returncode}（{'成功' if ok else '失败'}）",
    ]
    if stdout:
        parts.append(f"stdout:\n{stdout.strip()}")
    if stderr:
        parts.append(f"stderr:\n{stderr.strip()}")
    return "\n".join(parts)


check_dangerous = _check_dangerous


@tool
def gen_shell_commands_run(instruction: str) -> str:
    """
    根据用户指令生成并执行 shell 命令。
    高风险操作（rm、shutdown、重定向覆盖等）会被自动拦截；
    执行前会暂停图流程，向用户展示命令与说明，用户确认后才真正执行。
    """
    # 正常情况下，此工具的 tool_call 会被 tools_node 拦截并走人机确认流程；
    # 这里仅作为“脱离图流程被直接调用”时的兜底，不自动执行、要求使用图入口。
    command, explanation = gen_shell_command_spec(instruction)
    if not command:
        return "未能生成有效的 shell 命令"
    danger = check_dangerous(command)
    if danger:
        return f"已拦截高风险命令（{danger}）：{command}"
    return (
        "此工具需在图流程中运行以获得用户确认。\n"
        f"拟执行命令：{command}\n说明：{explanation or '(无)'}"
    )

tools = [
    check_weather,
    get_current_time,
    calculate,
    convert_currency,
    lookup_ip,
    read_file,
    pwd,
    gen_shell_commands_run,
]


tools_by_name = {tool.name: tool for tool in tools}
