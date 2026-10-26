"""Checks each service with one real page and prints what came back.

    python scripts/smoke_test.py page_elements path/to/file.pdf
    python scripts/smoke_test.py all path/to/file.pdf
Services: page_elements table_structure deepseek_ocr glm parse embed rerank graphic_elements llm milvus
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdfium2 as pdfium  # noqa: E402

from pipeline.config import cfg  # noqa: E402
from pipeline.extraction import (deepseek_ocr, glm, graphic_elements, ocr, page_elements, parse,  # noqa: E402
                                 table_structure)
from pipeline.extraction.render import crop  # noqa: E402
from pipeline.nim_client import get  # noqa: E402


def page_image(pdf: str, n: int = 0):
    doc = pdfium.PdfDocument(pdf)
    img = doc[min(n, len(doc) - 1)].render(scale=cfg()["render"]["dpi"] / 72).to_pil().convert("RGB")
    doc.close()
    return img


def gpu_mem() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def ready(service: str) -> None:
    url = cfg()["nims"][service]["url"]
    try:
        print(f"  ready: {get(url + cfg()['nims'][service].get('health', '/v1/health/ready'))}")
    except ValueError:  # vLLM's /health returns an empty 200
        print("  ready: ok")
    try:
        print(f"  models: {[m['id'] for m in get(url + '/v1/models')['data']]}")
    except Exception as e:  # not every NIM exposes /v1/models
        print(f"  models: n/a ({e})")


def run(service: str, pdf: str | None) -> None:
    print(f"\n== {service} ==  GPU memory {gpu_mem()}")
    t0 = time.time()
    if service == "page_elements":
        ready(service)
        dets = page_elements.detect([page_image(pdf)])[0]
        print(f"  {len(dets)} regions:", [(d.label, [round(x, 3) for x in d.box], round(d.score, 2)) for d in dets])
    elif service == "table_structure":
        ready(service)
        img = page_image(pdf)
        dets = page_elements.detect([img])[0]
        tables = [d for d in dets if d.label == "table"]
        target = crop(img, tables[0].box) if tables else img
        s = table_structure.detect([target])[0]
        print(f"  {'table crop' if tables else 'whole page (no table found)'}: "
              f"{sum(d.label == 'cell' for d in s)} cells, {sum(d.label == 'row' for d in s)} rows, "
              f"{sum(d.label == 'column' for d in s)} cols")
        if tables:
            r = ocr.read_tables([target], [s])[0]
            d = table_structure.mismatch(table_structure.grid_shape(s), r.text)
            print(f"  read by {r.extractor}, mismatch vs detected grid {d:.2f}\n{r.text[:1500]}")
    elif service == "deepseek_ocr":
        ready(service)
        text = table_structure.html_tables_to_markdown(deepseek_ocr.read(page_image(pdf), "markdown"))
        print(f"  {len(text)} chars; first 800:\n{text[:800]}")
    elif service == "glm":
        c = cfg()["nims"]["glm"]
        print(f"  {c['model']} at {c['url']}")
        text = glm.read(page_image(pdf), "Transcribe the first 10 lines of text on this page.")
        print(f"  {len(text)} chars:\n{text[:800]}")
    elif service == "parse":
        ready(service)
        blocks = parse.parse_page(page_image(pdf))
        print(f"  {len(blocks)} blocks, classes {sorted({b.cls for b in blocks})}\n{parse.page_text(blocks)[:800]}")
    elif service == "embed":
        ready(service)
        from pipeline.embedding import embed
        v = embed(["hello world", "a second passage"], "passage")
        q = embed(["hello?"], "query")
        print(f"  dim={len(v[0])} (config expects {cfg()['milvus']['dim']}); query dim={len(q[0])}")
    elif service == "rerank":
        ready(service)
        from pipeline.retrieval.rerank import rerank
        out = rerank("What is the capital of France?",
                     [{"text": "Berlin is in Germany."}, {"text": "Paris is the capital of France."}], 2)
        print("  ", [(o["text"], round(o["rerank_score"], 2)) for o in out])
    elif service == "graphic_elements":
        img = page_image(pdf)
        charts = [d for d in page_elements.detect([img])[0] if d.label == "chart"]
        target = crop(img, charts[0].box) if charts else img
        e = graphic_elements.detect([target])[0]
        print(f"  {'chart crop' if charts else 'whole page (no chart found)'}: {[(d.label, round(d.score, 2)) for d in e]}")
        print(f"  hint for the VLM: {graphic_elements.hints(e)!r}")
        if charts:
            r = ocr.read_charts([target], [graphic_elements.hints(e)])[0]
            print(f"  read by {r.extractor}:\n{r.text[:1500]}")
    elif service == "llm":
        from pipeline.retrieval.llm import chat
        print("  ", chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=20))
    elif service == "milvus":
        from pymilvus import MilvusClient
        c = MilvusClient(uri=cfg()["milvus"]["uri"])
        print("  collections:", c.list_collections())
    print(f"  ({time.time() - t0:.1f}s)  GPU memory {gpu_mem()}")


if __name__ == "__main__":
    svc = sys.argv[1]
    pdf = sys.argv[2] if len(sys.argv) > 2 else None
    order = ["milvus", "page_elements", "deepseek_ocr", "glm", "table_structure", "parse", "embed", "rerank",
             "graphic_elements", "llm"]
    for s in (order if svc == "all" else [svc]):
        try:
            run(s, pdf)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
