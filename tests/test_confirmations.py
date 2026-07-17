"""确认协议：按类分发、fs_write 危险路径闸门、审计脱敏、工具本体不落盘。"""

from __future__ import annotations

import pytest

from askanswer.confirmations import (
    FsWriteConfirmation,
    PaidApiConfirmation,
    get_confirmation_handler,
    redact_audit_args,
)
from askanswer.tools import validate_write_path


def test_handler_dispatch_by_class():
    assert get_confirmation_handler("shell") is not None
    assert isinstance(get_confirmation_handler("fs_write"), FsWriteConfirmation)
    assert isinstance(get_confirmation_handler("external_api_paid"), PaidApiConfirmation)
    assert get_confirmation_handler("none") is None
    assert get_confirmation_handler(None) is None


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_rsa",
        "/etc/passwd",
        ".env",
        "/home/user/project/.env",
        "server.pem",
    ],
)
def test_fs_write_rejects_sensitive_paths(path):
    assert validate_write_path(path, "data") is not None


def test_fs_write_rejects_oversize(tmp_path):
    big = "x" * (1024 * 1024 + 1)
    assert validate_write_path(str(tmp_path / "big.txt"), big) is not None


def test_fs_write_rejects_missing_parent(tmp_path):
    err = validate_write_path(str(tmp_path / "no-such-dir" / "f.txt"), "x")
    assert err is not None


def test_fs_write_allows_normal_file(tmp_path):
    assert validate_write_path(str(tmp_path / "ok.txt"), "hello") is None


def test_fs_write_plan_carries_error_for_bad_path():
    handler = FsWriteConfirmation()
    payload = handler.plan({"args": {"path": "/etc/passwd", "content": "x"}})
    assert payload.get("error")


def test_fs_write_apply_rejection_does_not_write(tmp_path):
    handler = FsWriteConfirmation()
    target = tmp_path / "f.txt"
    payload = handler.plan({"args": {"path": str(target), "content": "data"}})
    outcome = handler.apply(
        payload, False, {"args": {"path": str(target), "content": "data"}}
    )
    assert outcome.approved is False
    assert not target.exists()


def test_fs_write_apply_approval_writes(tmp_path):
    handler = FsWriteConfirmation()
    target = tmp_path / "f.txt"
    call = {"args": {"path": str(target), "content": "data"}}
    payload = handler.plan(call)
    outcome = handler.apply(payload, True, call)
    assert outcome.approved is True
    assert target.read_text(encoding="utf-8") == "data"
    # 审计只带 path，不带 content
    assert "content" not in outcome.audit_args


def test_fs_write_tool_body_never_writes(tmp_path):
    """不变量 7：工具本体只返回提示，绝不直接落盘。"""
    from askanswer.tools import write_file

    target = tmp_path / "f.txt"
    result = write_file.invoke({"path": str(target), "content": "data"})
    assert not target.exists()
    assert "确认" in result


def test_redact_masks_sensitive_keys():
    out = redact_audit_args(
        {"api_key": "sk-1", "Authorization": "Bearer x", "user_token": "t", "q": "hi"}
    )
    assert out["api_key"] == "***"
    assert out["Authorization"] == "***"
    assert out["user_token"] == "***"
    assert out["q"] == "hi"


def test_redact_does_not_overmatch():
    out = redact_audit_args({"monkey": "safe", "count": 3})
    assert out == {"monkey": "safe", "count": 3}


def test_redact_scrubs_emails_and_nesting():
    out = redact_audit_args(
        {"msg": "contact bob@example.com", "nested": {"password": "p", "list": ["a@b.co", 1]}}
    )
    assert "bob@example.com" not in out["msg"]
    assert out["nested"]["password"] == "***"
    assert out["nested"]["list"] == ["***", 1]


def test_paid_api_audit_paths_redacted():
    handler = PaidApiConfirmation()
    summary = handler.audit_args({"tool": "paid", "args": {"token": "x", "q": "hi"}})
    assert summary["args"]["token"] == "***"
    outcome = handler.apply(
        {"tool": "paid", "args": {"secret": "s"}}, False, {"name": "paid", "args": {}}
    )
    assert outcome.approved is False
    assert outcome.audit_args["args"]["secret"] == "***"
