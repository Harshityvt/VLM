"""Extraction pipeline over every PDF in data/pdfs. Safe to stop and re-run: it resumes from the
last finished page window of each PDF.

    python ingest.py                               # everything
    python ingest.py --max-pages 30 --dump chunks  # pilot: first 30 pages of each PDF, chunks to JSON
    python ingest.py --limit 10                    # first 10 PDFs only
    python ingest.py --reset                       # drop the collection and manifest, start over
"""
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipeline import manifest, vectordb
from pipeline.config import cfg, path
from pipeline.embedding import embed
from pipeline.extraction.assemble import page_chunks
from pipeline.extraction.render import iter_pages, page_count

WINDOW = 64  # pages rendered, extracted, embedded and committed together

log = logging.getLogger("ingest")
_bar_lock = threading.Lock()


def _windows(pdf: Path, did: str, start: int, end: int):
    batch = []
    for item in iter_pages(pdf, did, start, end):
        batch.append(item)
        if len(batch) == WINDOW:
            yield batch
            batch = []
    if batch:
        yield batch


def process(pdf: Path, pages_pool: ThreadPoolExecutor, bar: tqdm, max_pages: int | None,
            dump: Path | None) -> tuple[str, int]:
    did = manifest.doc_id(pdf)
    total = page_count(pdf)
    end = min(total, max_pages) if max_pages else total
    state = manifest.get(did) or {"status": "new", "next_page": 1, "chunks": 0, "seconds": 0.0}
    start = state["next_page"] or 1
    if start > end:
        with _bar_lock:
            bar.update(end)
        return "skipped", 0
    with _bar_lock:
        bar.update(start - 1)

    vectordb.delete_doc(did, from_page=start)   # drop anything from a window that didn't finish
    chunks_total, t0 = state["chunks"] or 0, time.time()
    dump_f = open(dump / f"{pdf.stem}.jsonl", "a", encoding="utf-8") if dump else None
    try:
        for window in _windows(pdf, did, start, end):
            per_page = list(pages_pool.map(lambda p: page_chunks(*p), window))
            chunks = [c for page in per_page for c in page]
            for c in chunks:
                c["doc_id"], c["file_name"] = did, pdf.name
            if chunks:
                for c, v in zip(chunks, embed([c["text"] for c in chunks], "passage")):
                    c["vector"] = v
                vectordb.insert(chunks)
            if dump_f:
                for c in chunks:
                    dump_f.write(json.dumps({k: v for k, v in c.items() if k != "vector"}, ensure_ascii=False) + "\n")
                dump_f.flush()
            chunks_total += len(chunks)
            next_page = window[-1][0] + 1
            status = "done" if next_page > end and end == total else "partial"
            manifest.update(did, str(pdf), status, total, next_page, chunks_total,
                            (state["seconds"] or 0) + time.time() - t0)
            with _bar_lock:
                bar.update(len(window))
                bar.set_postfix(chunks=chunks_total)
        return ("done" if end == total else "partial"), chunks_total
    except Exception as e:  # keep going with other PDFs; this one resumes from its last window next run
        log.exception("failed: %s", pdf)
        cur = manifest.get(did) or {"next_page": start, "chunks": chunks_total}
        manifest.update(did, str(pdf), "failed", total, cur["next_page"], cur["chunks"],
                        (state["seconds"] or 0) + time.time() - t0, error=f"{type(e).__name__}: {e}"[:2000])
        return "failed", chunks_total
    finally:
        if dump_f:
            dump_f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N PDFs")
    ap.add_argument("--max-pages", type=int, help="only the first N pages of each PDF (pilot)")
    ap.add_argument("--dump", type=Path, help="also write chunks to <dir>/<pdf>.jsonl for review")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    logs = path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logs / "ingest.log", encoding="utf-8")])

    if args.reset:
        manifest.reset()
    vectordb.ensure_collection(drop=args.reset)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in path("pdfs").rglob("*") if p.suffix.lower() == ".pdf")
    if args.limit:
        pdfs = pdfs[: args.limit]
    counts = [page_count(p) for p in pdfs]
    target = sum(min(n, args.max_pages) if args.max_pages else n for n in counts)
    print(f"{len(pdfs)} PDFs, {sum(counts):,} pages ({target:,} in this run) from {path('pdfs')}")

    ic = cfg()["ingest"]
    results = {"done": 0, "partial": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    with ThreadPoolExecutor(ic["page_concurrency"]) as pages_pool, \
            ThreadPoolExecutor(ic["pdf_concurrency"]) as pdf_pool, \
            tqdm(total=target, unit="page", smoothing=0.05) as bar:
        futs = [pdf_pool.submit(process, p, pages_pool, bar, args.max_pages, args.dump) for p in pdfs]
        for f in as_completed(futs):
            status, _ = f.result()
            results[status] += 1
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min: {results}")
    print("Manifest:", manifest.summary())
    if results["failed"]:
        print("Some PDFs failed; see logs/ingest.log. Re-run ingest.py to resume them.")


if __name__ == "__main__":
    main()
