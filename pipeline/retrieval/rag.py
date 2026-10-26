"""Retrieval pipeline, in diagram order:
User -> Guardrails(in) -> Query Processing -> Embedding -> Vector DB -> Reranking -> LLM
     -> Reflection (fail: back to Query Processing, max N) -> Guardrails(out) -> User
"""
import logging
import re
import time
from dataclasses import dataclass, field

from ..config import cfg
from ..embedding import embed
from ..vectordb import search
from . import answer, query_processing, reflection
from .rerank import rerank

log = logging.getLogger(__name__)


NOT_FOUND = ("I couldn't find a clear, supported answer to this in the documents. "
             "The closest passages are listed under Sources.")


@dataclass
class Result:
    answer: str
    chunks: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    blocked: str | None = None
    verified: bool = True
    timings: dict[str, float] = field(default_factory=dict)


def _content_key(text: str) -> str:
    """Chunk text minus its page-header prefix, punctuation and case, so repeated copies of the same
    passage (the manual reprints whole sections) compare equal."""
    if text.startswith("[") and "]\n" in text:
        text = text.split("]\n", 1)[1]
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def retrieve(queries: list[str], primary: str) -> list[dict]:
    rc = cfg()["retrieval"]
    hits_per_query = search(embed(queries, "query"), rc["top_k"])
    merged: dict[int, dict] = {}
    for hits in hits_per_query:
        for h in hits:
            if h["id"] not in merged or h["score"] > merged[h["id"]]["score"]:
                merged[h["id"]] = h
    # Collapse duplicate passages, keeping the best-scoring copy and noting the other pages it appears on.
    unique: dict[str, dict] = {}
    for h in sorted(merged.values(), key=lambda h: -h["score"]):
        key = _content_key(h["text"])
        if key in unique:
            unique[key].setdefault("also_on_pages", []).append(h["page"])
        else:
            unique[key] = h
    candidates = list(unique.values())[: rc["top_k"]]
    return rerank(primary, candidates, rc["rerank_top_n"])


class _Timer:
    """Accumulates seconds per pipeline stage, e.g. timings['rerank'] across retries."""

    def __init__(self, timings: dict[str, float]):
        self.timings = timings

    def __call__(self, stage: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.timings[stage] = timer.timings.get(stage, 0.0) + time.perf_counter() - self.t0

        return _Ctx()


def ask(question: str, history: list[dict] | None = None,
        use_guardrails: bool | None = None, use_reflection: bool | None = None) -> Result:
    """use_guardrails / use_reflection override config.yaml for this one question when given."""
    rc = cfg()["retrieval"]
    use_guardrails = rc["guardrails"] if use_guardrails is None else use_guardrails
    use_reflection = rc["reflection"] if use_reflection is None else use_reflection
    timings: dict[str, float] = {}
    timed = _Timer(timings)

    if use_guardrails:
        from . import guardrails
        with timed("guardrails_in"):
            allowed = guardrails.check_input(question)
        if not allowed:
            return Result(guardrails.REFUSAL, blocked="input", timings=timings)

    feedback, res = None, Result("", timings=timings)
    for attempt in range(rc["max_retries"] + 1):
        with timed("query_processing"):
            queries = query_processing.process(question, history, feedback)
        with timed("retrieve_rerank"):
            chunks = retrieve(queries, queries[0])
        context = answer.format_context(chunks)
        with timed("llm"):
            draft = answer.generate(queries[0], context, history)
        res = Result(draft, chunks, queries, res.reviews, timings=timings)
        if not use_reflection:
            break
        with timed("reflection"):
            verdict = reflection.review(question, context, draft)
        res.reviews.append(verdict)
        log.info("reflection attempt %d: %s", attempt + 1, verdict)
        if verdict["passed"]:
            break
        feedback = verdict.get("feedback") or "The answer was not supported by the retrieved passages."
    else:
        # every attempt failed review: don't hand the user an answer the judge rejected
        res.answer, res.verified = NOT_FOUND, False

    if use_guardrails:
        from . import guardrails
        with timed("guardrails_out"):
            allowed = guardrails.check_output(question, res.answer)
        if not allowed:
            return Result(guardrails.REFUSAL, res.chunks, res.queries, res.reviews, blocked="output",
                          timings=timings)
    return res
