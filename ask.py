"""Retrieval pipeline. Ask one question, or run without arguments for a chat loop.

    python ask.py "What was revenue in 2024?"
    python ask.py --show-chunks "..."
"""
import argparse
import logging

from pipeline import vectordb
from pipeline.retrieval.rag import ask


def show(res, show_chunks: bool) -> None:
    print("\n" + res.answer + "\n")
    if res.blocked:
        print(f"(blocked by {res.blocked} guardrail)")
        return
    print("Queries:", " | ".join(res.queries))
    for i, r in enumerate(res.reviews, 1):
        print(f"Reflection {i}: grounded={r.get('grounded')} answers={r.get('answers_question')} {r.get('feedback', '')}")
    print("Sources:")
    for c in res.chunks:
        also = c.get("also_on_pages") or []
        pages = "page" if len(also) == 1 else "pages"
        more = f"  (same text on {len(also)} more {pages}: {', '.join(map(str, sorted(also)[:5]))}{'…' if len(also) > 5 else ''})" if also else ""
        print(f"  {c['file_name']} p.{c['page']} [{c['element_type']}] rerank={c['rerank_score']:.2f}{more}")
        if show_chunks:
            print("    " + c["text"][:400].replace("\n", "\n    "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    vectordb.ensure_collection()

    if args.question:
        show(ask(args.question), args.show_chunks)
        return
    history: list[dict] = []
    while True:
        try:
            q = input("\nQuestion (empty to quit): ").strip()
        except EOFError:
            break
        if not q:
            break
        res = ask(q, history)
        show(res, args.show_chunks)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}]


if __name__ == "__main__":
    main()
