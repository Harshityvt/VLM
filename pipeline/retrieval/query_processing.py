"""'Query Processing': makes the question standalone, adds paraphrases, and applies Reflection feedback."""
import json
import re

from ..config import cfg
from .llm import chat

PROMPT = """You prepare search queries for a document retrieval system over a library of PDFs.

Conversation so far (may be empty):
{history}

Latest user question: {question}
{feedback}
Return JSON only, in this form:
{{"standalone": "<the question rewritten so it makes sense without the conversation>",
  "paraphrases": ["<alternative phrasing>", ...]}}
Give at most {n} paraphrases. Keep names, numbers and technical terms exactly as written."""


def process(question: str, history: list[dict] | None = None, feedback: str | None = None) -> list[str]:
    n = cfg()["retrieval"]["multi_query"]
    hist = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-6:])
    fb = (f"\nA previous answer failed review for this reason: {feedback}\n"
          "Write queries that would find the missing information.\n") if feedback else ""
    raw = chat([{"role": "user", "content": PROMPT.format(history=hist, question=question, feedback=fb, n=n)}],
               model=cfg()["llm"].get("fast_model"), temperature=0.0, max_tokens=400)
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        queries = [data["standalone"], *data.get("paraphrases", [])[:n]]
    except (AttributeError, KeyError, json.JSONDecodeError):
        queries = [question]
    seen, out = set(), []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out or [question]
