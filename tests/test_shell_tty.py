"""CLI 交互式 shell / 编辑器支持：TTY 检测、管道拒绝、/edit 解析。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from askanswer.cli.commands.edit import resolve_editor
from askanswer.tools import command_needs_tty, execute_shell_command

# ── command_needs_tty ──────────────────────────────────────────────


def test_command_needs_tty_editors():
    assert command_needs_tty("vim foo.py")
    assert command_needs_tty("nano notes.md")
    assert command_needs_tty("/usr/bin/nvim README.md")
    assert command_needs_tty("EDITOR=vim nano x")  # 跳过 env 赋值
    assert command_needs_tty("less /var/log/syslog")
    assert command_needs_tty("htop")


def test_command_needs_tty_non_interactive():
    assert not command_needs_tty("ls -la")
    assert not command_needs_tty("echo hello")
    assert not command_needs_tty("python -c 'print(1)'")
    assert not command_needs_tty("ssh host ls")  # ssh 允许非交互
    assert not command_needs_tty("")
    assert not command_needs_tty("FOO=bar")  # 只有赋值


# ── execute_shell_command 非 TTY 拒绝编辑器 ─────────────────────────


def test_execute_rejects_editor_without_tty():
    out = execute_shell_command("vim secret.py", shell=False, tty=False)
    assert "TTY" in out or "交互" in out
    assert "/edit" in out or "!vim" in out or "!" in out


def test_execute_normal_command_still_works():
    out = execute_shell_command("echo hello-tty-test", shell=True, tty=False)
    assert "hello-tty-test" in out
    assert "返回码：0" in out


def test_execute_tty_mode_inherits_stdio():
    """tty=True 时 Popen 不接 PIPE，wait 后只回报退出状态。"""
    fake = MagicMock()
    fake.returncode = 0
    fake.wait.return_value = 0
    with patch("askanswer.tools.subprocess.Popen", return_value=fake) as popen:
        out = execute_shell_command("vim foo.py", shell=True, tty=True)
    popen.assert_called_once()
    kwargs = popen.call_args.kwargs
    assert kwargs.get("stdout") is None
    assert kwargs.get("stderr") is None
    assert "返回码：0" in out
    assert "直通终端" in out
    fake.wait.assert_called()


# ── resolve_editor ─────────────────────────────────────────────────


def test_resolve_editor_prefers_askanswer_editor(monkeypatch):
    monkeypatch.setenv("ASKANSWER_EDITOR", "nano")
    monkeypatch.setenv("VISUAL", "vim")
    monkeypatch.setenv("EDITOR", "vi")
    assert resolve_editor() == ["nano"]


def test_resolve_editor_parses_args(monkeypatch):
    monkeypatch.setenv("ASKANSWER_EDITOR", "code --wait")
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert resolve_editor() == ["code", "--wait"]


def test_resolve_editor_falls_back_to_path(monkeypatch):
    for key in ("ASKANSWER_EDITOR", "VISUAL", "EDITOR"):
        monkeypatch.delenv(key, raising=False)

    def _which(name: str):
        return f"/usr/bin/{name}" if name == "vim" else None

    with patch("askanswer.cli.commands.edit.shutil.which", side_effect=_which):
        assert resolve_editor() == ["vim"]


def test_resolve_editor_none_when_missing(monkeypatch):
    for key in ("ASKANSWER_EDITOR", "VISUAL", "EDITOR"):
        monkeypatch.delenv(key, raising=False)
    with patch("askanswer.cli.commands.edit.shutil.which", return_value=None):
        assert resolve_editor() is None
