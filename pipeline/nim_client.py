"""Shared HTTP plumbing for NIM calls: pooled client, per-service concurrency, retry with backoff."""
import base64
import io
import logging
import random
import threading
import time

import httpx
from PIL import Image

from .config import cfg, nvidia_api_key

log = logging.getLogger(__name__)

_client = httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0),
                       limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
_semaphores: dict[str, threading.Semaphore] = {}
_lock = threading.Lock()

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def _sem(service: str) -> threading.Semaphore:
    with _lock:
        if service not in _semaphores:
            n = cfg()["nims"].get(service, {}).get("concurrency", 4)
            _semaphores[service] = threading.Semaphore(n)
        return _semaphores[service]


def post(service: str, url: str, payload: dict, hosted: bool = False, retries: int = 6,
         api_key: str | None = None) -> dict:
    """hosted=True sends the NVIDIA key; api_key sends another provider's key instead."""
    headers = {"Accept": "application/json"}
    if api_key or hosted:
        headers["Authorization"] = f"Bearer {api_key or nvidia_api_key()}"
    delay = 2.0
    for attempt in range(retries + 1):
        with _sem(service):
            try:
                r = _client.post(url, json=payload, headers=headers)
            except httpx.TransportError as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in RETRY_STATUS:
                    raise RuntimeError(f"{service} HTTP {r.status_code}: {r.text[:500]}")
                err = f"HTTP {r.status_code}"
        if attempt == retries:
            raise RuntimeError(f"{service} failed after {retries} retries: {err}")
        sleep = delay * (1 + random.random() * 0.25)
        log.warning("%s %s, retrying in %.1fs", service, err, sleep)
        time.sleep(sleep)
        delay = min(delay * 2, 60)


def get(url: str) -> dict:
    r = _client.get(url)
    r.raise_for_status()
    return r.json()


def image_data_url(img: Image.Image, quality: int = 90, max_b64_chars: int | None = None) -> str:
    """JPEG data URL. If max_b64_chars is set, shrink until the payload fits (hosted NIM cap)."""
    img = img.convert("RGB")
    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        if max_b64_chars is None or len(b64) <= max_b64_chars:
            return f"data:image/jpeg;base64,{b64}"
        if quality > 70:
            quality -= 10
        else:
            img = img.resize((max(1, int(img.width * 0.8)), max(1, int(img.height * 0.8))), Image.LANCZOS)
