# ``/model``：查看 / 热切换当前模型。
from __future__ import annotations

from ...audit import log_event
from ...load import current_model_label, set_model
from ..render import render_error
from ..theme import C


def handle_model_command(args: str, *, thread_id: str) -> None:
    """/model：无参数显示当前模型；带参数尝试切换模型。"""
    if not args:
        print()
        print(f"  {C.BOLD}Model{C.RESET}")
        print(f"   {C.DIM}current:{C.RESET} {current_model_label()}")
        print(f"   {C.DIM}usage:{C.RESET}   {C.CYAN}/model <name>{C.RESET} "
              f"{C.DIM}或{C.RESET} {C.CYAN}/model <provider:name>{C.RESET}")
        print()
        return

    try:
        # set_model 是热替换，所有已 import 的 model 引用都会自动指向新模型
        label = set_model(args)
    except Exception as exc:
        render_error(f"模型切换失败: {exc}")
        return
    log_event(
        kind="model_swap",
        thread_id=thread_id,
        model_label=label,
        args_summary=args,
        immediate=True,
    )

    print()
    print(f"  {C.GREEN}✓ 已切换模型:{C.RESET} {C.BOLD}{label}{C.RESET}")
    print()
