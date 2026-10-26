"""Local web UI for the RAG pipeline.

    .venv\\Scripts\\python ui.py          # opens http://127.0.0.1:7861
    .venv\\Scripts\\python ui.py --port 8080 --no-browser

Needs the Docker stack running (docker compose up -d). Listens on 127.0.0.1 only.
"""
import argparse
import logging
import re
import threading
import time
import webbrowser

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import manifest, vectordb
from pipeline.config import ROOT, cfg, path
from pipeline.retrieval.rag import ask

log = logging.getLogger("ui")
app = FastAPI(title="NeMo PDF RAG")
UI_DIR = ROOT / "ui"

SERVICES = [  # (label, config key) for the status bar, in diagram order
    ("Page Elements", "page_elements"), ("Table Structure", "table_structure"), ("DeepSeek-OCR", "deepseek_ocr"),
    ("Parse", "parse"), ("Embedding", "embed"), ("Reranking", "rerank"),
]


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = []
    guardrails: bool = True
    reflection: bool = True


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/status")
def status():
    nims = cfg()["nims"]
    out = []
    with httpx.Client(timeout=2.0) as c:
        for label, key in SERVICES:
            try:
                ok = c.get(nims[key]['url'] + nims[key].get('health', '/v1/health/ready')).status_code == 200
            except httpx.HTTPError:
                ok = False
            out.append({"name": label, "ok": ok, "where": "local"})
    try:
        vectordb.ensure_collection()
        # count(*) rather than collection stats: stats only include flushed segments, so a fresh ingest shows 0
        rows = int(vectordb.client().query(cfg()["milvus"]["collection"], filter="",
                                           output_fields=["count(*)"])[0]["count(*)"])
        out.append({"name": "Milvus", "ok": True, "where": "local"})
    except Exception:  # Milvus down or collection missing
        rows = 0
        out.append({"name": "Milvus", "ok": False, "where": "local"})
    out.append({"name": "Graphic Elements", "ok": None, "where": "hosted"})
    out.append({"name": "GLM-4.6V", "ok": None, "where": "hosted", "model": nims["glm"]["model"]})
    out.append({"name": "LLM", "ok": None, "where": "hosted", "model": cfg()["llm"]["model"]})
    docs = manifest.summary()
    return {"services": out, "chunks": rows,
            "documents": sum(v["docs"] for v in docs.values()),
            "pages": sum(v["pages_done"] for v in docs.values())}


def _source(c: dict) -> dict:
    text = c["text"]
    header = ""
    if text.startswith("[") and "]\n" in text:
        header, text = text[1:].split("]\n", 1)
    m = re.match(r"data/pages/([0-9a-f]{40})/(p\d{4,5}\.jpg)$", c.get("page_image", ""))
    return {
        "file_name": c["file_name"], "doc_id": c["doc_id"], "page": c["page"],
        "element_type": c["element_type"], "extractor": c.get("extractor", ""), "bbox": c.get("bbox") or None,
        "rerank_score": round(float(c.get("rerank_score", 0.0)), 3),
        "also_on_pages": sorted(c.get("also_on_pages") or []),
        "header": header, "text": text,
        "image": f"/pages/{m.group(1)}/{m.group(2)}" if m else None,
    }


@app.post("/api/ask")
def api_ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        res = ask(req.question.strip(), [t.model_dump() for t in req.history][-8:],
                  use_guardrails=req.guardrails, use_reflection=req.reflection)
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(502, f"{type(e).__name__}: {e}") from e
    return {
        "answer": res.answer, "blocked": res.blocked, "verified": res.verified,
        "queries": res.queries, "reviews": res.reviews,
        "timings": {k: round(v, 2) for k, v in res.timings.items()},
        "total_seconds": round(time.perf_counter() - t0, 2),
        "sources": [_source(c) for c in res.chunks],
        "reflection": req.reflection, "guardrails": req.guardrails,
    }


@app.get("/pages/{doc_id}/{name}")
def page_image(doc_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{40}", doc_id) or not re.fullmatch(r"p\d{4,5}\.jpg", name):
        raise HTTPException(404)
    f = path("pages") / doc_id / name
    if not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"RAG UI running at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
