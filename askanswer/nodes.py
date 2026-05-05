# 父图前置节点：意图分类（understand）和事后评估（sorcery）。
# 设计原则：先用本地正则/关键词做高置信度分类，避免每次都调用 LLM；
# 仅当本地启发不确定时才回退到 LLM 来判定 intent。
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .load import model
from .state import SearchState


# 系统支持的四种意图类别
_INTENTS = {"file_read", "sql", "search", "chat"}
# 用于本地识别 file_read 意图的常见后缀，新增类型时记得在这里同步
_FILE_EXTENSIONS = (
    "txt", "md", "markdown", "rst",
    "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
    "csv", "tsv", "xlsx", "xls",
    "pdf", "docx", "pptx",
    "html", "htm", "xml", "svg",
    "py", "pyi", "ipynb",
    "js", "jsx", "ts", "tsx",
    "go", "rs", "java", "kt", "swift", "rb", "php",
    "c", "h", "cpp", "hpp", "cc", "cs",
    "sh", "bash", "zsh", "ps1",
    "sql", "log",
)
# 文件路径的正则：支持 Windows 盘符、相对路径、HOME 缩写、各种引号包裹等场景
_FILE_PATH_RE = re.compile(
    r"""(?ix)
    (?:
        ["'`“”‘’]?
        (
            (?:[a-z]:[\\/]|\.{1,2}[\\/]|[\\/]|~/)?
            [^\s"'`“”‘’<>|]+
            \.(?:%s)
        )
        ["'`“”‘’]?
    )
    """ % "|".join(_FILE_EXTENSIONS)
)
# 命中即视为 SQL 意图的关键词
_SQL_KEYWORDS = (
    "sql",
    "数据库",
    "数据表",
    "表结构",
    "查询表",
    "查表",
    "建表",
    "postgres",
    "mysql",
    "sqlite",
    "建库",
)
# 直接出现 select/insert/... 的 SQL 语句样式，作为额外信号
_SQL_RE = re.compile(r"(?is)\b(select|insert|update|delete)\b.+\b(from|into|set|where)\b")
# 命中即视为联网搜索意图的关键词（含中英文）
_SEARCH_KEYWORDS = (
    "联网",
    "搜索",
    "搜一下",
    "查一下",
    "查找",
    "最新",
    "最近",
    "今天",
    "今日",
    "现在",
    "实时",
    "新闻",
    "价格",
    "股价",
    "汇率",
    "天气",
    "官网",
    "资料",
    "search",
    "latest",
    "recent",
    "today",
    "current",
    "realtime",
    "news",
    "price",
    "stock",
    "weather",
    "official",
)
# 配合文件路径出现时，更确认是 file_read 意图的动作词
_FILE_READ_KEYWORDS = (
    "读",
    "读取",
    "分析",
    "打开",
    "查看",
    "文件",
    "read",
    "analyze",
    "open",
    "view",
    "file",
)
# 闲聊起手词；输入以这些开头时倾向 chat
_CHAT_STARTERS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "hey",
    "解释",
    "说明",
    "总结",
    "翻译",
    "改写",
    "写一段",
    "帮我写",
    "如何",
    "怎么",
    "为什么",
)


def _extract_file_path(text: str) -> str:
    """从一句话中提取出第一个看起来像文件路径的子串，找不到返回空串。"""
    match = _FILE_PATH_RE.search(text)
    if not match:
        return ""
    # 去掉两端可能的引号
    return match.group(1).strip().strip("\"'`“”‘’")


def _local_intent(user_message: str, *, fallback: bool = False) -> dict | None:
    """高置信度的本地意图分类器。

    返回 None 表示“说不清楚”，由调用方决定是否走 LLM；当 ``fallback=True`` 时
    则强制给出一个尽量合理的 intent，作为 LLM 失败时的兜底。
    """
    text = str(user_message or "").strip()
    lowered = text.lower()
    file_path = _extract_file_path(text)

    # 1) 既有文件路径又有 read/分析 等动作词 → 视为 file_read
    if file_path and any(word in lowered for word in _FILE_READ_KEYWORDS):
        return _normalize_intent(
            {
                "intent": "file_read",
                "file_path": file_path,
                "search_query": "",
                "understanding": f"读取或分析本地文件：{file_path}",
            },
            text,
        )

    # 2) 命中 SQL 关键词或直接写了 SQL 语句 → SQL 意图
    if any(keyword in lowered for keyword in _SQL_KEYWORDS) or _SQL_RE.search(text):
        return _normalize_intent(
            {
                "intent": "sql",
                "file_path": "",
                "search_query": "",
                "understanding": text,
            },
            text,
        )

    # 3) 命中搜索关键词 → 走联网搜索
    if any(keyword in lowered for keyword in _SEARCH_KEYWORDS):
        return _normalize_intent(
            {
                "intent": "search",
                "file_path": "",
                "search_query": text,
                "understanding": text,
            },
            text,
        )

    # fallback 路径：LLM 走不通也得给一个意图，否则父图无法继续
    if fallback:
        # 有路径但没动作词时仍优先视为 file_read
        if file_path:
            return _normalize_intent(
                {
                    "intent": "file_read",
                    "file_path": file_path,
                    "search_query": "",
                    "understanding": f"读取或分析本地文件：{file_path}",
                },
                text,
            )
        # 默认按 chat 处理；很长且非问句倾向于让模型搜索一下
        intent = "chat"
        if len(text) > 80 and "?" not in text and "？" not in text:
            intent = "search"
        elif any(lowered.startswith(starter) for starter in _CHAT_STARTERS):
            intent = "chat"
        elif not text:
            intent = "chat"
        return _normalize_intent(
            {
                "intent": intent,
                "file_path": file_path if intent == "file_read" else "",
                "search_query": text if intent == "search" else "",
                "understanding": text,
            },
            text,
        )

    # 以闲聊起手词开头：直接 chat
    if any(lowered.startswith(starter) for starter in _CHAT_STARTERS):
        return _normalize_intent(
            {
                "intent": "chat",
                "file_path": "",
                "search_query": "",
                "understanding": text,
            },
            text,
        )

    # 其它情况：交给 LLM 判断
    return None


def _parse_json_object(text: str) -> dict:
    """容错地把 LLM 输出解析为 dict：去掉 ```json 围栏、容忍前后多余文本。"""
    raw = str(text or "").strip()
    # 去除 markdown 代码块围栏
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 容错：从首个 { 到末尾 } 截一段再尝试
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("intent JSON must be an object")
    return data


def _normalize_intent(fields: dict, user_message: str) -> dict:
    """把任何来源的 intent dict 归一化成标准格式（统一字段、补默认值）。"""
    intent = str(fields.get("intent") or "").lower().strip()
    # 不在白名单中的 intent 一律退回 search 兜底
    if intent not in _INTENTS:
        intent = "search"

    # file_read 必须带路径；如果 LLM 没给，再尝试从原文提取一次
    file_path = str(fields.get("file_path") or "").strip()
    if intent == "file_read" and not file_path:
        file_path = _extract_file_path(user_message)

    # search_query 仅对 search 意图有意义，其它意图清空
    search_query = str(fields.get("search_query") or "").strip()
    if intent == "search" and not search_query:
        search_query = user_message
    elif intent != "search":
        search_query = ""

    return {
        "intent": intent,
        "file_path": file_path if intent == "file_read" else "",
        "search_query": search_query,
        "understanding": str(fields.get("understanding") or user_message).strip(),
    }


def _intent_from_llm(user_message: str) -> dict:
    """调用 LLM 做意图分类，要求其返回严格 JSON。"""
    prompt = (
        f'分析用户的查询："{user_message}"\n\n'
        "请判断用户意图属于以下哪一类：\n"
        "- file_read：要求读取或分析本地文件（通常会给出路径或文件名，"
        "如 /tmp/a.txt、./data.csv、report.md）\n"
        "- sql：要求查询数据库、编写 SQL、分析表结构、统计数据库数据\n"
        "- chat：闲聊、常识性问题，或你已知可直接回答、不需要联网\n"
        "- search：需要联网搜索获取实时、最新或不确定的信息\n\n"
        "只输出一个 JSON 对象，不要 Markdown、不要解释。字段如下：\n"
        "{\n"
        '  "intent": "file_read|sql|chat|search",\n'
        '  "file_path": "仅 file_read 时填写具体路径，否则为空字符串",\n'
        '  "search_query": "仅 search 时填写最佳关键词，否则为空字符串",\n'
        '  "understanding": "对用户需求的简要总结"\n'
        "}\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    return _normalize_intent(_parse_json_object(response.content), user_message)


def understand_query_node(state: SearchState) -> dict:
    """父图第一个节点：把用户最新消息分类为 file_read/sql/chat/search。"""
    user_message = state["messages"][-1].content
    # 优先用本地分类器
    fields = _local_intent(user_message)
    if fields is None:
        # 不确定时再调用 LLM；LLM 也失败就走 fallback 强制给个意图，避免流程卡死
        try:
            fields = _intent_from_llm(user_message)
        except Exception:
            fields = _local_intent(user_message, fallback=True)

    intent = fields["intent"]
    search_query = fields["search_query"]
    file_path = fields["file_path"]
    understanding = fields["understanding"]

    # 为 CLI 进度条/日志生成一句易读的说明
    if intent == "file_read":
        hint = f"识别为读文件：{file_path or '(未能提取路径)'}"
    elif intent == "sql":
        hint = "识别为数据库/SQL 问题，转交 sql_query 工具"
    elif intent == "chat":
        hint = "识别为闲聊/常识问题，直接回答"
    else:
        hint = f"识别为搜索，关键词：{search_query}"

    # 注意：节点返回的是“部分状态”字典，会被 LangGraph 合并到 SearchState
    return {
        "user_query": understanding,
        "search_query": search_query,
        "file_path": file_path,
        "intent": intent,
        "retry_count": 0,
        "step": "understood",
        "messages": [AIMessage(content=hint)],
    }


def sorcery_answer_node(state: SearchState) -> dict:
    """父图第三个节点：评估当前答案是否令人满意，仅 search 路径会触发重试。"""
    intent = state.get("intent", "search")
    final_answer = state.get("final_answer", "")

    # 只有 search 路径允许评估并要求重试；其它意图直接通过
    if intent != "search":
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    # 已经重试过一次就放行，避免无穷循环 / 无止境消耗 token
    retry_count = state.get("retry_count", 0)
    if retry_count >= 1:
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    # 让 LLM 给当前答案打分；约定严格输出格式以便正则解析
    prompt = f"""请评估当前答案是否足够好。
    用户问题：{state.get('user_query', '')}
    当前搜索词：{state.get('search_query', '')}
    当前搜索结果：{state.get('search_results', '')}
    当前答案：{final_answer}

    如果答案已经足够好，请严格输出：
    评分：pass
    新搜索词：

    如果答案不够好，需要重新搜索，请严格输出：
    评分：re_search
    新搜索词：[更精准的新搜索词]
    """
    response = model.invoke([SystemMessage(content=prompt)])
    response_text = response.content

    # 命中 re_search 标记且给出了新关键词，则触发一次重搜
    if "评分：re_search" in response_text and "新搜索词：" in response_text:
        new_search_query = response_text.split("新搜索词：", 1)[1].strip()
        if new_search_query:
            # 用一条合成的 HumanMessage 让下一轮 _answer_node 主动调用 tavily_search
            retry_msg = (
                f"前一次回答不够理想，请调用 tavily_search 工具用以下关键词重新搜索后再作答：{new_search_query}"
            )
            return {
                "search_query": new_search_query,
                "retry_count": retry_count + 1,
                "step": "retry_search",
                "messages": [
                    AIMessage(content=f"当前答案不够理想，改为搜索：{new_search_query}"),
                    HumanMessage(content=retry_msg),
                ],
            }

    # 默认通过
    return {
        "final_answer": final_answer,
        "step": "completed",
        "messages": [AIMessage(content=final_answer)],
    }
