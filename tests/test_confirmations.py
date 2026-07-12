"""确认协议：按 class 分发、fs_write 危险路径全拒、审计脱敏、plan→gate→interrupt→apply 四步。"""

from __future__ import annotations

import pytest

from askanswer import confirmations as C


# ── handler 分发 ─────────────────────────────────────────────────────────

def test_supported_classes():
    assert C.supported_confirmation_classes() == frozenset(
        {"shell", "fs_write", "external_api_paid"}
    )


@pytest.mark.parametrize(
    "clazz,expected",
    [
        ("shell", C.ShellConfirmation),
        ("fs_write", C.FsWriteConfirmation),
        ("external_api_paid", C.PaidApiConfirmation),
    ],
)
def test_get_handler_dispatches_by_class(clazz, expected):
    assert isinstance(C.get_confirmation_handler(clazz), expected)


@pytest.mark.parametrize("clazz", [None, "", "none", "unknown"])
def test_get_handler_none_for_unmapped(clazz):
    assert C.get_confirmation_handler(clazz) is None


# ── fs_write 危险路径全拒 ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/authorized_keys",
        "/tmp/secret.pem",
        "/tmp/id_rsa",
        "/tmp/service.key",
        "~/.netrc",
    ],
)
def test_fs_write_plan_rejects_sensitive_paths(path):
    handler = C.FsWriteConfirmation()
    payload = handler.plan({"name": "write_file", "args": {"path": path, "content": "x"}})
    assert "error" in payload  # 敏感路径在 plan 阶段即被拦截


def test_fs_write_plan_rejects_oversize(monkeypatch, tmp_path):
    # 把上限压到 10 字节，写 11 字节应被拒
    monkeypatch.setattr("askanswer.tools._WRITE_FILE_MAX_BYTES", 10)
    handler = C.FsWriteConfirmation()
    payload = handler.plan(
        {"name": "write_file", "args": {"path": str(tmp_path / "f.txt"), "content": "x" * 11}}
    )
    assert "error" in payload


def test_fs_write_apply_denied_does_not_write(tmp_path):
    target = tmp_path / "out.txt"
    handler = C.FsWriteConfirmation()
    payload = handler.plan(
        {"name": "write_file", "args": {"path": str(target), "content": "hello"}}
    )
    outcome = handler.apply(payload, decision=False,
                            tool_call={"args": {"path": str(target), "content": "hello"}})
    assert outcome.approved is False
    assert not target.exists()  # 拒绝 → 不落盘（不变量 7）


def test_fs_write_apply_approved_writes(tmp_path):
    target = tmp_path / "out.txt"
    handler = C.FsWriteConfirmation()
    call = {"name": "write_file", "args": {"path": str(target), "content": "hello"}}
    payload = handler.plan(call)
    outcome = handler.apply(payload, decision=True, tool_call=call)
    assert outcome.approved is True
    assert target.read_text(encoding="utf-8") == "hello"


# ── shell gate 拦截危险命令 ──────────────────────────────────────────────

def test_shell_gate_blocks_dangerous():
    handler = C.ShellConfirmation()
    blocked = handler.gate({"command": "rm -rf /", "explanation": ""})
    assert blocked is not None
    assert "rm" in blocked


def test_shell_gate_allows_safe():
    handler = C.ShellConfirmation()
    assert handler.gate({"command": "ls -la", "explanation": ""}) is None


def test_shell_apply_rechecks_edited_command():
    """用户批准时把命令改成危险的，apply 必须重新拦截（双重检查）。"""
    handler = C.ShellConfirmation()
    outcome = handler.apply(
        {"command": "ls"}, decision={"approve": True, "command": "rm -rf /"},
        tool_call={},
    )
    assert outcome.approved is False
    assert outcome.error is not None


def test_shell_interrupt_payload_type():
    handler = C.ShellConfirmation()
    payload = handler.interrupt_payload({"command": "ls", "explanation": "list", "instruction": "看看"})
    assert payload["type"] == "confirm_shell"
    assert payload["command"] == "ls"


# ── 决策解析 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "decision,expected",
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("y", True),
        ("1", True),
        ("no", False),
        (None, False),
        ({"approve": True}, True),
        ({"value": "yes"}, True),
    ],
)
def test_parse_approval(decision, expected):
    assert C.parse_approval(decision) is expected


def test_parse_decision_preserves_edited_command():
    approved, command = C.parse_decision({"approve": True, "command": "echo hi"}, "fallback")
    assert approved is True
    assert command == "echo hi"


# ── 审计脱敏 ─────────────────────────────────────────────────────────────

def test_redact_sensitive_keys():
    out = C.redact_audit_args({"api_key": "sk-secret", "token": "abc", "path": "/x"})
    assert out["api_key"] == "***"
    assert out["token"] == "***"
    assert out["path"] == "/x"  # 非敏感 key 原样保留


def test_redact_email_in_values():
    out = C.redact_audit_args({"note": "reach me at a@b.com please"})
    assert "a@b.com" not in out["note"]
    assert "***" in out["note"]


def test_redact_nested_and_depth_capped():
    deep = {"l1": {"l2": {"l3": {"l4": {"l5": "x"}}}}}
    out = C.redact_audit_args(deep)
    # 达到最大深度后整体折叠为 ***，不无限递归
    assert C._REDACTED in repr(out)


def test_redact_list_of_dicts():
    out = C.redact_audit_args([{"password": "p"}, {"ok": "v"}])
    assert out[0]["password"] == "***"
    assert out[1]["ok"] == "v"
