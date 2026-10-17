"""主图拓扑：扁平 7 节点 + golden Mermaid；导入 graph 不触发持久化。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from askanswer.graph import create_search_assistant


def _mermaid() -> str:
    app = create_search_assistant(InMemorySaver())
    return app.get_graph().draw_mermaid()


# ── 拓扑：单层扁平 7 节点（D1 / 架构文档 4.2） ─────────────────────────

GOLDEN = Path(__file__).parent / "golden" / "graph.mmd"
NODES = ("route", "clarify", "reason", "plan_approvals", "ask_approval", "act", "review")


def test_flat_graph_has_exactly_seven_nodes():
    app = create_search_assistant(InMemorySaver())
    nodes = set(app.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == set(NODES)


def test_flat_graph_edges():
    mermaid = _mermaid()
    assert "__start__ --> route;" in mermaid
    assert "route --> clarify;" in mermaid
    assert "clarify --> reason;" in mermaid
    assert "reason -.-> act;" in mermaid
    assert "reason -.-> plan_approvals;" in mermaid
    assert "reason -.-> review;" in mermaid
    assert "plan_approvals --> ask_approval;" in mermaid
    assert "ask_approval -.-> act;" in mermaid
    assert "act --> reason;" in mermaid
    assert "review -.-> __end__;" in mermaid
    assert "review -.-> reason;" in mermaid


def test_graph_matches_golden_mermaid():
    """拓扑变更必须显式更新 tests/golden/graph.mmd（防止无意改图）。"""
    assert _mermaid().strip() == GOLDEN.read_text(encoding="utf-8").strip()


def test_no_intent_forking_in_graph():
    """主图不按 intent 分叉 —— 不应出现 file_read/sql/chat 等意图节点。"""
    mermaid = _mermaid()
    for intent in ("file_read", "sql", "chat", "helix", "research", "decision"):
        assert f"{intent}(" not in mermaid


# ── 导入/构图不触发持久化（不变量 3） ────────────────────────────────────

def test_building_graph_does_not_create_db(monkeypatch, tmp_path):
    db_path = tmp_path / "should_not_exist.db"
    monkeypatch.setenv("ASKANSWER_DB_PATH", str(db_path))
    # 用 InMemorySaver 构图不应触碰 SqliteSaver / state.db
    create_search_assistant(InMemorySaver())
    assert not db_path.exists()


def test_graph_module_import_has_no_persistence_singleton():
    """import graph 不应实例化 persistence 单例（get_store 只在 CLI 入口触发）。"""
    import askanswer.storage.store as persistence

    # 干净导入路径下，单例应仍为 None（本测试不调用 get_store）
    assert persistence._singleton is None


# ── 干净子进程里的导入副作用（比同进程断言更强） ─────────────────────────

def test_import_graph_has_no_persistence_side_effect(tmp_path):
    """不变量 3：import graph 不得触发 persistence 初始化（不建 state.db）。"""
    db = tmp_path / "side" / "state.db"
    code = (
        "import askanswer.graph, pathlib, sys; "
        f"sys.exit(1 if pathlib.Path(r'{db}').exists() else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"ASKANSWER_DB_PATH": str(db), "PATH": _path_env()},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-500:]


def test_import_helix_mcp_has_no_persistence_side_effect(tmp_path):
    db = tmp_path / "side" / "state.db"
    code = (
        "import askanswer.interfaces.mcp_server.helix_server, pathlib, sys; "
        f"sys.exit(1 if pathlib.Path(r'{db}').exists() else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"ASKANSWER_DB_PATH": str(db), "PATH": _path_env()},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-500:]


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "")
