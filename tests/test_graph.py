"""主图拓扑：固定的三步骨架（不变量 1）+ 导入 graph 不触发持久化。"""

from __future__ import annotations

import subprocess
import sys

from langgraph.checkpoint.memory import InMemorySaver

from askanswer.graph import create_search_assistant


def _mermaid() -> str:
    app = create_search_assistant(InMemorySaver())
    return app.get_graph().draw_mermaid()


# ── 拓扑 golden（对齐 spec §2.1：START → understand → answer → sorcery → (END|answer)）──

def test_parent_graph_nodes():
    mermaid = _mermaid()
    for node in ("understand", "answer", "sorcery"):
        assert f"{node}(" in mermaid, f"缺少节点 {node}"


def test_parent_graph_edges():
    mermaid = _mermaid()
    # 固定骨架的实边
    assert "__start__ --> understand;" in mermaid
    assert "understand --> answer;" in mermaid
    assert "answer --> sorcery;" in mermaid
    # sorcery 的条件分支：结束 或 回到 answer 重试
    assert "sorcery -.-> __end__;" in mermaid
    assert "sorcery -.-> answer;" in mermaid


def test_no_intent_forking_in_parent_graph():
    """不变量 1：父图不按 intent 分叉 —— 不应出现 file_read/sql/chat 等意图节点。"""
    mermaid = _mermaid()
    for intent in ("file_read", "sql", "chat", "helix", "research", "decision"):
        assert f"{intent}(" not in mermaid


def test_confirm_plan_not_in_parent_graph():
    """confirm_plan 属于 answer 内的 react 子图，不应出现在父图顶层。"""
    mermaid = _mermaid()
    assert "confirm_plan(" not in mermaid


# ── 导入/构图不触发持久化（不变量 3） ────────────────────────────────────

def test_building_graph_does_not_create_db(monkeypatch, tmp_path):
    db_path = tmp_path / "should_not_exist.db"
    monkeypatch.setenv("ASKANSWER_DB_PATH", str(db_path))
    # 用 InMemorySaver 构图不应触碰 SqliteSaver / state.db
    create_search_assistant(InMemorySaver())
    assert not db_path.exists()


def test_graph_module_import_has_no_persistence_singleton():
    """import graph 不应实例化 persistence 单例（get_persistence 只在 CLI 入口触发）。"""
    import askanswer.persistence as persistence

    # 干净导入路径下，单例应仍为 None（本测试不调用 get_persistence）
    assert persistence._singleton is None


# ── react 子图结构 ───────────────────────────────────────────────────────

def test_react_subgraph_has_confirm_plan():
    from askanswer.react import build_react_subgraph

    nodes = set(build_react_subgraph().get_graph().nodes)
    assert {"answer", "tools", "confirm_plan"} <= nodes


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
        "import askanswer.helix_mcp, pathlib, sys; "
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
