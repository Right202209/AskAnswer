from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage

from .graph import create_search_assistant

HEADER_LINE = "=" * 56
SECTION_LINE = "-" * 56
PROMPT = "你 > "


def run_query(app: object, query: str, thread_id: str) -> str:
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer

    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        if isinstance(content, str):
            return content

    return "未生成答案。"


def interactive_loop(app: object) -> int:
    thread_id = str(uuid.uuid4())

    print(HEADER_LINE)
    print("AskAnswer 命令行助手")
    print("输入问题后回车；输入 exit / quit / q 退出。")
    print(HEADER_LINE)

    while True:
        try:
            query = input(PROMPT).strip()
        except EOFError:
            print("\n再见。")
            return 0
        except KeyboardInterrupt:
            print("\n已取消，输入 exit 退出。")
            continue

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("再见。")
            return 0

        try:
            answer = run_query(app, query, thread_id)
        except Exception as exc:
            print(f"\n{SECTION_LINE}\n运行失败\n{SECTION_LINE}\n{exc}\n")
            continue

        print(f"\n{SECTION_LINE}\n回答\n{SECTION_LINE}\n{answer}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AskAnswer 命令行工具")
    parser.add_argument("question", nargs="?", help="要提问的内容")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        app = create_search_assistant()
    except Exception as exc:
        print(f"{SECTION_LINE}\n初始化失败\n{SECTION_LINE}\n{exc}")
        return 1

    if args.question:
        thread_id = str(uuid.uuid4())
        try:
            answer = run_query(app, args.question, thread_id)
        except Exception as exc:
            print(f"{SECTION_LINE}\n运行失败\n{SECTION_LINE}\n{exc}")
            return 1
        print(f"{SECTION_LINE}\n回答\n{SECTION_LINE}\n{answer}")
        return 0

    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())
