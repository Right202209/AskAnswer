"""Intent 注册表：8 个 handler 齐全、classify_local 回退规则、normalize 字段清洗。"""

from __future__ import annotations

import pytest

from askanswer.intents import get_intent_registry
from askanswer.intents.base import IntentClassification

EXPECTED_HANDLERS = {
    "file_read", "sql", "helix", "decision", "math", "research", "search", "chat",
}


def test_eight_handlers_registered():
    assert get_intent_registry().names() == EXPECTED_HANDLERS


def test_handlers_sorted_by_priority():
    handlers = get_intent_registry().handlers()
    priorities = [h.priority for h in handlers]
    assert priorities == sorted(priorities)


# ── classify_local 本地分类 ──────────────────────────────────────────────

def test_classify_local_math():
    result = get_intent_registry().classify_local("计算 12 * 7")
    assert result is not None
    assert result.intent == "math"


def test_classify_local_arithmetic_expression():
    result = get_intent_registry().classify_local("3 + 4 * 2")
    assert result is not None
    assert result.intent == "math"


def test_classify_local_file_path():
    result = get_intent_registry().classify_local("查看 /tmp/data.csv")
    assert result is not None
    assert result.intent == "file_read"
    assert result.file_path.endswith("data.csv")


def test_classify_local_no_match_returns_none():
    # 不带 fallback 时无本地规则命中 → None（留给 LLM 分类）。
    # 输入刻意避开 search 关键词、chat 起手词、文件路径与算式。
    assert get_intent_registry().classify_local("帮我起一个名字") is None


def test_classify_local_fallback_produces_result():
    result = get_intent_registry().classify_local("随便聊聊", fallback=True)
    assert result is not None
    assert result.intent in EXPECTED_HANDLERS


# ── normalize 字段清洗 ───────────────────────────────────────────────────

def test_normalize_non_search_clears_search_query():
    reg = get_intent_registry()
    fields = IntentClassification(intent="chat", search_query="不该保留")
    out = reg.normalize(fields, "你好")
    assert out.search_query == ""


def test_normalize_non_file_read_clears_file_path():
    reg = get_intent_registry()
    fields = IntentClassification(intent="chat", file_path="/tmp/x.txt")
    out = reg.normalize(fields, "你好")
    assert out.file_path == ""


def test_normalize_search_defaults_query_to_message():
    reg = get_intent_registry()
    fields = IntentClassification(intent="search", search_query="")
    out = reg.normalize(fields, "上海天气")
    assert out.search_query == "上海天气"


def test_normalize_file_read_extracts_path_from_message():
    reg = get_intent_registry()
    fields = IntentClassification(intent="file_read", file_path="")
    out = reg.normalize(fields, "帮我读 ./report.md")
    assert out.file_path.endswith("report.md")


def test_normalize_unknown_intent_falls_back_to_search():
    reg = get_intent_registry()
    fields = IntentClassification(intent="nonexistent")
    out = reg.normalize(fields, "查一下")
    assert out.intent == "search"


def test_normalize_understanding_defaults_to_message():
    reg = get_intent_registry()
    fields = IntentClassification(intent="chat", understanding="")
    out = reg.normalize(fields, "原始消息")
    assert out.understanding == "原始消息"


@pytest.mark.parametrize("name,fallback_name", [(None, "search"), ("nonexistent", "search")])
def test_get_unknown_intent_falls_back_to_search(name, fallback_name):
    handler = get_intent_registry().get(name)
    assert handler.name == fallback_name


def test_get_normalizes_case_and_whitespace():
    # get() 对名字做 lower().strip()，大小写/空白包裹仍命中 search
    assert get_intent_registry().get("  SEARCH  ").name == "search"


# ── dict 入参归一（LLM 结构化输出常是裸 dict，normalize 需吃 dict） ───────

def test_normalize_clears_cross_intent_fields():
    result = get_intent_registry().normalize(
        {"intent": "chat", "search_query": "leftover", "file_path": "/tmp/x"},
        "你好",
    )
    assert result.intent == "chat"
    assert result.search_query == ""  # 非 search 清空
    assert result.file_path == ""  # 非 file_read 清空


def test_normalize_unknown_intent_coerced_to_search():
    result = get_intent_registry().normalize({"intent": "bogus"}, "问题")
    assert result.intent == "search"


# ── classify_local 带 fallback 时对真实文件命中 file_read ────────────────

def test_classify_local_fallback_path_detection(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2", encoding="utf-8")
    result = get_intent_registry().classify_local(f"帮我看看 {f}", fallback=True)
    assert result is not None
    assert result.intent == "file_read"
    assert result.file_path


# ── tool_tags 委派到 handler.bundle_tags ─────────────────────────────────

def test_tool_tags_delegate_to_handler():
    tags = get_intent_registry().tool_tags("research")
    assert isinstance(tags, frozenset)
    assert "research" in tags
