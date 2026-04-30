import json
import re

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime

from .load import model, tavily_client
from .schema import ContextSchema, normalize_context
from .sqlagent.sql_agent import extract_sql_answer, run_sql_agent
from .state import SearchState
from .tools import tools_by_name


_INTENTS = {"file_read", "sql", "search", "chat"}
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
_SQL_RE = re.compile(r"(?is)\b(select|insert|update|delete)\b.+\b(from|into|set|where)\b")
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
    match = _FILE_PATH_RE.search(text)
    if not match:
        return ""
    return match.group(1).strip().strip("\"'`“”‘’")


def _local_intent(user_message: str, *, fallback: bool = False) -> dict | None:
    """Fast local classifier for high-confidence requests.

    Returns None when the message is ambiguous enough to deserve the LLM.
    """
    text = str(user_message or "").strip()
    lowered = text.lower()
    file_path = _extract_file_path(text)

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

    if fallback:
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

    return None


def _parse_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("intent JSON must be an object")
    return data


def _normalize_intent(fields: dict, user_message: str) -> dict:
    intent = str(fields.get("intent") or "").lower().strip()
    if intent not in _INTENTS:
        intent = "search"

    file_path = str(fields.get("file_path") or "").strip()
    if intent == "file_read" and not file_path:
        file_path = _extract_file_path(user_message)

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
    user_message = state["messages"][-1].content
    fields = _local_intent(user_message)
    if fields is None:
        try:
            fields = _intent_from_llm(user_message)
        except Exception:
            fields = _local_intent(user_message, fallback=True)

    intent = fields["intent"]
    search_query = fields["search_query"]
    file_path = fields["file_path"]
    understanding = fields["understanding"]

    if intent == "file_read":
        hint = f"识别为读文件：{file_path or '(未能提取路径)'}"
    elif intent == "sql":
        hint = "识别为数据库/SQL 问题，转交 SQL agent"
    elif intent == "chat":
        hint = "识别为闲聊/常识问题，直接回答"
    else:
        hint = f"识别为搜索，关键词：{search_query}"

    return {
        "user_query": understanding,
        "search_query": search_query,
        "file_path": file_path,
        "intent": intent,
        "retry_count": 0,
        "step": "understood",
        "messages": [AIMessage(content=hint)],
    }


def file_read_node(state: SearchState) -> dict:
    path = (state.get("file_path") or "").strip()
    if not path:
        msg = "未能识别要读取的文件路径，请明确指出文件路径后重试。"
        return {
            "final_answer": msg,
            "step": "completed",
            "messages": [AIMessage(content=msg)],
        }
    try:
        content = tools_by_name["read_file"].invoke({"path": path})
    except Exception as exc:
        content = f"读取 `{path}` 失败：{exc}"
    return {
        "final_answer": str(content),
        "step": "completed",
        "messages": [AIMessage(content=str(content))],
    }


def sql_agent_node(state: SearchState, runtime: Runtime[ContextSchema]) -> dict:
    try:
        sql_messages = run_sql_agent(
            list(state["messages"]),
            context=normalize_context(getattr(runtime, "context", None)),
        )
        final_answer = extract_sql_answer(sql_messages)
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": sql_messages,
        }
    except Exception as exc:
        message = f"SQL agent 执行失败：{exc}"
        return {
            "final_answer": message,
            "step": "completed",
            "messages": [AIMessage(content=message)],
        }


def tavily_search_node(state: SearchState) -> dict:
    search_query = state["search_query"]
    try:
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )

        results = response.get("results", [])
        tavily_answer = (response.get("answer") or "").strip()

        search_results = f"查询关键词：{search_query}\n\n"
        if tavily_answer:
            search_results += f"Tavily 摘要：{tavily_answer}\n\n"
        search_results += "搜索结果（Top 5）：\n\n"

        if results:
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                url = result.get("url", "#")
                content = result.get("content", "无内容摘要")
                score = result.get("score", 0.0)

                search_results += (
                    f"{i}. **{title}** (相关度: {score:.3f})\n"
                    f"   链接: {url}\n"
                    f"   {content[:280]}{'...' if len(content) > 280 else ''}\n\n"
                )
        else:
            search_results += "未找到任何搜索结果。\n"

        return {
            "search_results": search_results.strip(),
            "step": "searched",
            "messages": [AIMessage(content="搜索完成~ 正在整理答案...")],
        }
    except Exception as e:
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content=" 搜索遇到问题...")],
        }


def sorcery_answer_node(state: SearchState) -> dict:
    intent = state.get("intent", "search")
    final_answer = state.get("final_answer", "")

    # 只有 search 路径允许改写搜索词重跑；file_read / chat 直接通过
    if intent != "search":
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    retry_count = state.get("retry_count", 0)
    if retry_count >= 1:
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

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

    if "评分：re_search" in response_text and "新搜索词：" in response_text:
        new_search_query = response_text.split("新搜索词：", 1)[1].strip()
        if new_search_query:
            return {
                "search_query": new_search_query,
                "retry_count": retry_count + 1,
                "step": "retry_search",
                "messages": [
                    AIMessage(content=f"当前答案不够理想，改为搜索：{new_search_query}")
                ],
            }

    return {
        "final_answer": final_answer,
        "step": "completed",
        "messages": [AIMessage(content=final_answer)],
    }
