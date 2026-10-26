"""'NeMo Retriever Reranking': scores (query, chunk) pairs and keeps the best few."""
from ..config import cfg
from ..nim_client import post


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []
    c = cfg()["nims"]["rerank"]
    payload = {"model": c["model"], "query": {"text": query},
               "passages": [{"text": ch["text"]} for ch in chunks], "truncate": "END"}
    res = post("rerank", f"{c['url']}/v1/ranking", payload)
    ranked = sorted(res["rankings"], key=lambda r: r["logit"], reverse=True)[:top_n]
    return [{**chunks[r["index"]], "rerank_score": r["logit"]} for r in ranked]
