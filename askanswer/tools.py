# 内置工具集合：天气、时间、计算、汇率、IP、读文件、联网搜索、shell 执行等。
# 这里定义的每个 @tool 函数都会通过 registry.py 注册到统一工具表，
# 然后由 react 子图按 intent 暴露给 LLM。
import ast
import difflib
import json
import operator
import os
import platform
import re
import shlex
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from markitdown import MarkItDown

from .load import model, openweather_api_key, tavily_client


@tool
def check_weather(city: str) -> str:
    """查询指定城市的实时天气。参数 city 为城市名，例如 Beijing。"""

    # API Key 缺失时给出明确提示，而不是抛异常 —— 更友好且不影响图流程
    if not openweather_api_key:
        return "未配置 OPENWEATHER_API_KEY"

    # 拼接 OpenWeather One Call 接口的查询串：公制单位 + 中文描述
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
def read_file(path: str) -> str:
    """
    读取不同类型的文件并进行智能解析。
    参数: path (str): 文件路径
    返回: str: AI 对文件内容的分析结果
    """
    # 安全闸门：拦截敏感文件、超大文件、不存在或非普通文件的路径，避免被 prompt
    # 注入诱导去读 SSH/凭据/历史数据库或喂入畸形大文件拖垮进程。
    error = _validate_read_path(path)
    if error:
        return error
    # 先用 markitdown 把任何格式（pdf/docx/xlsx/...）转成统一的 Markdown 文本
    data = markitdown(path)
    # 再交给 LLM 做语义分析；结构化文件说字段，代码文件解释逻辑
    resp = model.invoke(f"分析这个文件,解释内容，"
                 f"包括结构、用途和关键信息。"
                 f"如果是代码就解释逻辑，如果是数据文件就说明字段，如果是二进制文件就说明用途。"
                 f"以下是文件:\n{data}")
    return resp.content

def markitdown(file_path: str) -> str:
    """把任意支持的文件类型转成 Markdown 文本，作为 read_file 的预处理步骤。"""
    md = MarkItDown(enable_plugins=True)
    result = md.convert(file_path)
    return result.text_content


# read_file 安全限制：默认 10 MB，可通过环境变量 ASKANSWER_READ_FILE_MAX_BYTES 调整
_READ_FILE_MAX_BYTES = int(os.environ.get("ASKANSWER_READ_FILE_MAX_BYTES") or 10 * 1024 * 1024)

# 命中即拒绝读取的敏感文件名/扩展（不区分大小写）。
# 设计：宁可误拒少数同名普通文件，也不能让 LLM 把凭据/密钥读进上下文。
_SENSITIVE_PATH_RE = re.compile(
    r"""(?ix)
    (?:^|/)(?:
        \.env(?:\.[^/]*)?$            # .env / .env.local / .env.production
      | id_(?:rsa|dsa|ed25519|ecdsa)  # SSH 私钥
      | authorized_keys$
      | known_hosts$
      | credentials$                  # AWS / gcloud
      | \.pgpass$
      | \.netrc$
      | \.htpasswd$
      | shadow$
      | wallet\.dat$
    )
    | \.(?:pem|key|p12|pfx|kdbx|gpg|asc|jks|keystore)$
    """,
)


def _validate_read_path(path: str) -> str | None:
    """读文件前的安全校验。返回错误字符串表示拒绝；None 表示放行。"""
    raw = (path or "").strip()
    if not raw:
        return "未提供文件路径"
    try:
        # 只解析符号链接到真实路径用于敏感检测；下游仍用原始路径让 markitdown 处理
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return f"路径解析失败：{exc}"
    if not resolved.exists():
        return f"文件不存在：{raw}"
    if not resolved.is_file():
        return f"路径不是普通文件：{raw}"
    # 同时检查原始路径与解析后路径，避免符号链接绕过
    for candidate in {raw, str(resolved)}:
        if _SENSITIVE_PATH_RE.search(candidate):
            return f"已拒绝读取敏感文件：{Path(candidate).name}"
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return f"获取文件大小失败：{exc}"
    if size > _READ_FILE_MAX_BYTES:
        return (
            f"文件过大（{size} 字节 > 上限 {_READ_FILE_MAX_BYTES}）；"
            f"如需读取，调整 ASKANSWER_READ_FILE_MAX_BYTES 环境变量。"
        )
    return None


# write_file 安全限制：默认 1 MB，可通过环境变量 ASKANSWER_WRITE_FILE_MAX_BYTES 调整。
# 写入内容会随规划一起进入 checkpoint（tool_call args），上限刻意比读取小一个量级。
_WRITE_FILE_MAX_BYTES = int(os.environ.get("ASKANSWER_WRITE_FILE_MAX_BYTES") or 1024 * 1024)
# fs_write 确认框里 diff / 预览的最大行数，超出截断
_WRITE_DIFF_MAX_LINES = 200
_WRITE_PREVIEW_MAX_LINES = 40


def validate_write_path(path: str, content: str) -> str | None:
    """写文件前的安全校验。返回错误字符串表示拒绝；None 表示放行。

    与 ``_validate_read_path`` 同源的敏感路径正则 —— 能骗 LLM 读凭据的路径，
    同样不允许被写入（覆盖 authorized_keys / .env 比读取更危险）。
    在“规划确认内容”与“用户批准后执行”两个时机都会被调用（对齐 shell 的双重检查）。
    """
    raw = (path or "").strip()
    if not raw:
        return "未提供文件路径"
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return f"路径解析失败：{exc}"
    # 同时检查原始路径与解析后路径，避免符号链接绕过
    for candidate in {raw, str(resolved)}:
        if _SENSITIVE_PATH_RE.search(candidate):
            return f"已拒绝写入敏感文件：{Path(candidate).name}"
    if resolved.exists() and not resolved.is_file():
        return f"路径已存在且不是普通文件：{raw}"
    parent = resolved.parent
    if not parent.is_dir():
        # 刻意不自动创建目录：目录级副作用应该由用户显式完成
        return f"父目录不存在：{parent}"
    size = len(content.encode("utf-8", errors="replace"))
    if size > _WRITE_FILE_MAX_BYTES:
        return (
            f"写入内容过大（{size} 字节 > 上限 {_WRITE_FILE_MAX_BYTES}）；"
            f"如需写入，调整 ASKANSWER_WRITE_FILE_MAX_BYTES 环境变量。"
        )
    return None


def describe_write(path: str, content: str) -> dict:
    """生成 fs_write 确认框所需的展示信息：是否覆盖、大小、diff 或内容预览。

    只返回“派生信息”—— 完整 content 已随 tool_call args 存在 checkpoint 的消息
    历史里，这里不再复制一份，避免 pending_confirmations 膨胀。
    """
    target = Path((path or "").strip()).expanduser().resolve()
    exists = target.is_file()
    try:
        size_before = target.stat().st_size if exists else 0
    except OSError:
        size_before = 0
    diff = ""
    if exists:
        try:
            old_text = target.read_text(encoding="utf-8", errors="replace")
            diff_lines = list(
                difflib.unified_diff(
                    old_text.splitlines(),
                    content.splitlines(),
                    fromfile=f"{path}（当前）",
                    tofile=f"{path}（写入后）",
                    lineterm="",
                )
            )
            if len(diff_lines) > _WRITE_DIFF_MAX_LINES:
                omitted = len(diff_lines) - _WRITE_DIFF_MAX_LINES
                diff_lines = diff_lines[:_WRITE_DIFF_MAX_LINES] + [
                    f"…（diff 已截断，省略 {omitted} 行）"
                ]
            diff = "\n".join(diff_lines) if diff_lines else "（内容无变化）"
        except Exception:
            # 旧文件读不出来（如二进制）就退化为内容预览
            diff = ""
    preview = ""
    if not diff:
        lines = content.splitlines()
        preview = "\n".join(lines[:_WRITE_PREVIEW_MAX_LINES])
        if len(lines) > _WRITE_PREVIEW_MAX_LINES:
            preview += f"\n…（预览已截断，共 {len(lines)} 行）"
    return {
        "exists": exists,
        "size_before": size_before,
        "size_after": len(content.encode("utf-8", errors="replace")),
        "diff": diff,
        "preview": preview,
    }


@tool
def write_file(path: str, content: str) -> str:
    """
    把文本内容写入指定文件（新建或覆盖已有文件）。
    敏感路径（.env、SSH 密钥、凭据等）会被自动拦截；
    写入前会暂停图流程，向用户展示 diff / 内容预览，用户确认后才真正落盘。
    参数：path 目标文件路径；content 要写入的完整文本内容。
    """
    # 正常情况下，此工具的 tool_call 会被 tools_node 拦截并走 fs_write 人机确认流程；
    # 这里仅作为“脱离图流程被直接调用”时的兜底，不落盘、要求使用图入口。
    error = validate_write_path(path, content)
    if error:
        return error
    return (
        "此工具需在图流程中运行以获得用户确认。\n"
        f"拟写入：{path}（{len(content.encode('utf-8', errors='replace'))} 字节）"
    )


@tool
def tavily_search(query: str) -> str:
    """联网搜索实时或最新信息。参数 query 为搜索关键词，返回 Top 5 网页摘要与 Tavily 综合回答。"""

    try:
        # search_depth=basic 速度快；include_answer 让 Tavily 顺带给一个综合摘要
        response = tavily_client.search(
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


# 高风险命令模式列表：在“生成命令”和“用户编辑命令”两个时机都会被检查
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
    """检查命令是否命中危险模式，命中返回中文描述，否则返回 None。"""
    for pattern, desc in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return desc
    return None


def _clean_command(raw: str) -> str:
    """清理 LLM 输出里偶尔残留的 ``` 代码块围栏与开头的 $ 提示符。"""
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
    """根据自然语言指令生成（命令, 解释）二元组。

    会把当前操作系统类型告诉 LLM，让它生成对应平台可直接执行的单条命令。
    """
    # 把当前 OS 信息拼到 prompt 里，避免 mac/linux/windows 命令混用
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
    # 逐行扫描；同时兼容中文和英文冒号
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
    # 兜底：如果两行格式都没匹配上，就把整段当成命令再清理一遍
    command = _clean_command(command or text)
    return command, explanation


# 需要真实 TTY 的全屏/交互程序。bang 模式会继承 stdio；管道捕获模式会直接拒绝。
# 不含 ssh（常用于非交互远程命令，靠超时兜底）。
_TTY_PROGRAMS = frozenset({
    "vi", "vim", "nvim", "nano", "emacs", "emacsclient",
    "ed", "joe", "micro", "helix", "hx", "ne", "tilde",
    "less", "more", "most", "view", "hexedit",
    "top", "htop", "btop", "iotop", "atop",
    "man", "info",
    "tmux", "screen",
})


def _first_program(command: str) -> str | None:
    """取出命令串里的首个可执行程序名（忽略前导 env 赋值）。"""
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    i = 0
    # 跳过 FOO=bar 形式的环境变量赋值
    while i < len(parts) and "=" in parts[i] and not parts[i].startswith("-"):
        i += 1
    if i >= len(parts):
        return None
    return Path(parts[i]).name.lower()


def command_needs_tty(command: str) -> bool:
    """命令是否几乎必然需要真实终端（nano/vim/less/top 等）。"""
    prog = _first_program(command)
    return prog is not None and prog in _TTY_PROGRAMS


def execute_shell_command(
    command: str,
    shell: bool = False,
    *,
    tty: bool = False,
) -> str:
    """执行 shell 命令并把 stdout/stderr/返回码包成易读的文本返回。

    ``tty=True``：继承当前进程的 stdin/stdout/stderr，无超时——供 CLI
    ``!vim`` / ``!nano`` / ``/edit`` 等交互式编辑器用。
    默认 ``tty=False``：捕获输出、30s 超时，给 agent / 非交互路径用。
    """
    # 管道捕获模式下拒绝全屏程序，避免 30s 假死后再超时
    if not tty and command_needs_tty(command):
        return (
            f"命令需要交互式终端（TTY），当前执行模式不支持：{command}\n"
            f"请在 CLI 中使用 `!{command}` 或 `/edit <path>`。"
        )

    popen_args: str | list[str]
    # shell=False 时用 shlex 拆分参数，避免命令注入；shell=True 时直接交给 /bin/sh
    if shell:
        popen_args = command
    else:
        try:
            popen_args = shlex.split(command)
        except ValueError as exc:
            return f"命令解析失败：{exc}\n原始命令：{command}"

    try:
        if tty:
            # 继承 TTY：用户直接在编辑器里操作，直到退出
            process = subprocess.Popen(popen_args, shell=shell)
            try:
                process.wait()
            except KeyboardInterrupt:
                # Ctrl-C：转发给子进程并等待收尸
                try:
                    process.send_signal(signal.SIGINT)
                except Exception:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return (
                    f"命令：{command}\n"
                    f"返回码：{process.returncode}（被中断）\n"
                    f"(交互式程序，输出已直通终端)"
                )
            stdout, stderr = "", ""
        else:
            process = subprocess.Popen(
                popen_args,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # 30s 超时：避免误执行长时间运行的命令导致整个图卡住
            stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        # 超时后必须 kill + 收尸，否则会有僵尸进程
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
    if tty:
        parts.append("(交互式程序，输出已直通终端)")
        return "\n".join(parts)
    # 输出截断：每路 stream 各 64 KB 上限，防止单条命令的大输出把 prompt 撑爆。
    if stdout:
        parts.append(f"stdout:\n{_truncate_stream(stdout.strip(), 'stdout')}")
    if stderr:
        parts.append(f"stderr:\n{_truncate_stream(stderr.strip(), 'stderr')}")
    return "\n".join(parts)


# 单路 stdout/stderr 的文本上限；超出则截断并附说明，避免拖垮 LLM prompt 与 checkpoint
_SHELL_OUTPUT_MAX_BYTES = int(os.environ.get("ASKANSWER_SHELL_OUTPUT_MAX_BYTES") or 64 * 1024)


def _truncate_stream(text: str, label: str) -> str:
    """对 shell 子进程单路输出做硬截断；命中上限时附带省略字符数提示。"""
    if len(text) <= _SHELL_OUTPUT_MAX_BYTES:
        return text
    omitted = len(text) - _SHELL_OUTPUT_MAX_BYTES
    return f"{text[:_SHELL_OUTPUT_MAX_BYTES]}\n[{label} 已截断，省略 {omitted} 字符]"


# 对外暴露的危险检查别名（_react_internals 与 cli 都会用到）
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

# 工具集合（旧式 list 形式），仍保留以便外部直接 import；新增工具优先通过 registry 注册
tools = [
    check_weather,
    get_current_time,
    calculate,
    convert_currency,
    lookup_ip,
    read_file,
    write_file,
    tavily_search,
    pwd,
    gen_shell_commands_run,
]


# 工具名 → 工具对象的字典：常用于按名称查找
tools_by_name = {tool.name: tool for tool in tools}
