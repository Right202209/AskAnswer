"""图级：拓扑 golden 快照、import graph 不触发 persistence 副作用。"""

from __future__ import annotations

import subprocess
import sys


def test_topology_nodes_and_edges():
    from askanswer.graph import create_search_assistant

    mermaid = create_search_assistant().get_graph().draw_mermaid()
    for node in ("understand", "answer", "sorcery"):
        assert node in mermaid
    # 固定拓扑：understand -> answer -> sorcery -> (END | answer)
    assert "understand --> answer" in mermaid
    assert "answer --> sorcery" in mermaid


def test_react_subgraph_has_confirm_plan():
    from askanswer.react import build_react_subgraph

    nodes = set(build_react_subgraph().get_graph().nodes)
    assert {"answer", "tools", "confirm_plan"} <= nodes


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
