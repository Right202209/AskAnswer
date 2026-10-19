"""审计脱敏：敏感 key 整体打码、字符串内 email 打码、递归深度封顶。"""

from __future__ import annotations

from askanswer.observability import redaction as R

# ── 审计脱敏 ─────────────────────────────────────────────────────────────

def test_redact_sensitive_keys():
    out = R.redact_audit_args({"api_key": "sk-secret", "token": "abc", "path": "/x"})
    assert out["api_key"] == "***"
    assert out["token"] == "***"
    assert out["path"] == "/x"  # 非敏感 key 原样保留


def test_redact_email_in_values():
    out = R.redact_audit_args({"note": "reach me at a@b.com please"})
    assert "a@b.com" not in out["note"]
    assert "***" in out["note"]


def test_redact_nested_and_depth_capped():
    deep = {"l1": {"l2": {"l3": {"l4": {"l5": "x"}}}}}
    out = R.redact_audit_args(deep)
    # 达到最大深度后整体折叠为 ***，不无限递归
    assert R._REDACTED in repr(out)


def test_redact_list_of_dicts():
    out = R.redact_audit_args([{"password": "p"}, {"ok": "v"}])
    assert out[0]["password"] == "***"
    assert out[1]["ok"] == "v"
