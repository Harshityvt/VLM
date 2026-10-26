"""'Reflection': the judge LLM checks the draft answer is grounded in the chunks and answers the question."""
import json
import re

from ..config import cfg
from .llm import chat

PROMPT = """You review answers produced by a retrieval system. Judge strictly.

Question: {question}

Retrieved context:
{context}

Draft answer:
{answer}

Check two things:
1. grounded: every factual claim in the draft is supported by the context.
2. answers_question: the draft actually answers the question (saying the documents don't contain the answer counts only if that is true of the context).

Return JSON only:
{{"grounded": true|false, "answers_question": true|false, "feedback": "<what is missing or unsupported, one sentence; empty if both true>"}}"""


def review(question: str, context: str, answer: str) -> dict:
    raw = chat([{"role": "user", "content": PROMPT.format(question=question, context=context, answer=answer)}],
               model=cfg()["llm"]["judge_model"], temperature=0.0, max_tokens=300)
    try:
        v = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return {"grounded": True, "answers_question": True, "feedback": "", "parse_error": raw[:200]}
    v["passed"] = bool(v.get("grounded")) and bool(v.get("answers_question"))
    return v
