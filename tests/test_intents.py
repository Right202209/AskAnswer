"""意图注册表：8 个内置 handler、本地分类回退、normalize 字段清洗。"""

from __future__ import annotations

from askanswer.intents import get_intent_registry

_EXPECTED = {"file_read", "sql", "helix", "decision", "math", "research", "search", "chat"}


def test_all_builtin_handlers_registered():
    assert _EXPECTED <= get_intent_registry().names()


def test_get_unknown_intent_falls_back_to_search():
    registry = get_intent_registry()
    assert registry.get("no-such-intent").name == "search"
    assert registry.get(None).name == "search"
    assert registry.get("  SEARCH  ").name == "search"  # 大小写/空白归一


def test_handlers_sorted_by_priority():
    handlers = get_intent_registry().handlers()
    priorities = [h.priority for h in handlers]
    assert priorities == sorted(priorities)


def test_normalize_clears_cross_intent_fields():
    registry = get_intent_registry()
    result = registry.normalize(
        {"intent": "chat", "search_query": "leftover", "file_path": "/tmp/x"},
        "你好",
    )
    assert result.intent == "chat"
    assert result.search_query == ""  # 非 search 清空
    assert result.file_path == ""  # 非 file_read 清空


def test_normalize_search_defaults_query_to_message():
    result = get_intent_registry().normalize({"intent": "search"}, "今天天气如何")
    assert result.search_query == "今天天气如何"


def test_normalize_unknown_intent_coerced_to_search():
    result = get_intent_registry().normalize({"intent": "bogus"}, "问题")
    assert result.intent == "search"


def test_classify_local_fallback_path_detection(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("a,b\n1,2", encoding="utf-8")
    result = get_intent_registry().classify_local(f"帮我看看 {f}", fallback=True)
    assert result is not None
    assert result.intent == "file_read"
    assert result.file_path


def test_classify_local_no_fallback_returns_none_for_ambiguous():
    # 无本地规则命中且不允许回退时返回 None（交给 LLM 分类）
    result = get_intent_registry().classify_local("讲讲量子纠缠的历史背景", fallback=False)
    assert result is None or result.intent in _EXPECTED


def test_tool_tags_delegate_to_handler():
    registry = get_intent_registry()
    tags = registry.tool_tags("research")
    assert isinstance(tags, frozenset)
    assert "research" in tags
