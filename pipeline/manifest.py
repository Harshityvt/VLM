"""SQLite record of extraction progress per PDF, down to the last finished page window,
so an interrupted run resumes mid-document instead of starting over."""
import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from .config import path

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(path("manifest"), check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS docs (
        doc_id TEXT PRIMARY KEY, file TEXT, status TEXT, pages INTEGER, next_page INTEGER,
        chunks INTEGER, error TEXT, seconds REAL, updated REAL)""")
    return c


_db = _conn()


def doc_id(pdf: Path) -> str:
    """sha1 of the file contents, so renamed or moved files are still recognised."""
    h = hashlib.sha1()
    with open(pdf, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(did: str) -> dict | None:
    with _lock:
        row = _db.execute("SELECT status, pages, next_page, chunks, seconds FROM docs WHERE doc_id=?",
                          (did,)).fetchone()
    if not row:
        return None
    return dict(zip(("status", "pages", "next_page", "chunks", "seconds"), row))


def update(did: str, file: str, status: str, pages: int, next_page: int, chunks: int,
           seconds: float, error: str | None = None) -> None:
    with _lock:
        _db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?,?,?)",
                    (did, file, status, pages, next_page, chunks, error, seconds, time.time()))
        _db.commit()


def reset() -> None:
    with _lock:
        _db.execute("DELETE FROM docs")
        _db.commit()


def summary() -> dict:
    with _lock:
        rows = _db.execute("SELECT status, COUNT(*), SUM(next_page - 1), SUM(chunks) FROM docs GROUP BY status").fetchall()
    return {s: {"docs": n, "pages_done": p or 0, "chunks": c or 0} for s, n, p, c in rows}
