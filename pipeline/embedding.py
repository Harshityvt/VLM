"""'NeMo Retriever Embedding': same NIM for ingest (passage) and retrieval (query)."""
from .config import cfg
from .nim_client import post


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    c = cfg()["nims"]["embed"]
    out: list[list[float]] = []
    for i in range(0, len(texts), c["batch_size"]):
        payload = {"model": c["model"], "input": texts[i:i + c["batch_size"]], "input_type": input_type,
                   "encoding_format": "float", "truncate": "END", "modality": "text"}
        res = post("embed", f"{c['url']}/v1/embeddings", payload)
        out += [d["embedding"] for d in sorted(res["data"], key=lambda d: d["index"])]
    return out
