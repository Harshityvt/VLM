"""Answer generation prompt for the 'LLM' box."""
from .llm import chat

SYSTEM = """You answer questions using only the numbered context passages from the user's PDF library.
Cite every claim with the passage's tag, e.g. [report.pdf p.12].
If the passages do not contain the answer, say so plainly instead of guessing.
Tables are given as Markdown; read them carefully."""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        tag = f"[{c['file_name']} p.{c['page']}]"
        parts.append(f"{tag} ({c['element_type']})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate(question: str, context: str, history: list[dict] | None = None) -> str:
    msgs = [{"role": "user", "content": f"{SYSTEM}\n\nContext:\n{context}"},
            {"role": "assistant", "content": "Understood. I will answer only from these passages, with citations."}]
    msgs += (history or [])[-6:]
    msgs.append({"role": "user", "content": question})
    return chat(msgs)
