from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import HumanMessage

try:
    from AskAnswer.graph import create_search_assistant
except ImportError:
    from graph import create_search_assistant


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

    while True:
        try:
            query = input("请输入问题（输入 exit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            return 0

        try:
            answer = run_query(app, query, thread_id)
        except Exception as exc:
            print(f"\n运行失败：{exc}\n")
            continue

        print(f"\n{answer}\n")



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
        print(f"初始化失败：{exc}")
        return 1

    if args.question:
        thread_id = str(uuid.uuid4())
        try:
            answer = run_query(app, args.question, thread_id)
        except Exception as exc:
            print(f"运行失败：{exc}")
            return 1
        print(answer)
        return 0

    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())