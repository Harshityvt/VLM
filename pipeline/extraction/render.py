"""'Pages as Images': renders PDF pages to JPEG, streaming so a 10,000-page PDF never sits in memory at once."""
from collections.abc import Iterator
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from ..config import cfg, path


def page_count(pdf: Path) -> int:
    doc = pdfium.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def iter_pages(pdf: Path, did: str, start: int = 1, end: int | None = None
               ) -> Iterator[tuple[int, Image.Image, Path]]:
    """Yields (page number starting at 1, PIL image, saved jpeg path) for pages start..end inclusive.
    pdfium is not thread-safe, so rendering stays on the calling thread."""
    rc = cfg()["render"]
    out_dir = path("pages") / did
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = rc["dpi"] / 72
    doc = pdfium.PdfDocument(str(pdf))
    try:
        last = min(end or len(doc), len(doc))
        for n in range(start, last + 1):
            page = doc[n - 1]
            img = page.render(scale=scale).to_pil().convert("RGB")
            page.close()
            dest = out_dir / f"p{n:05d}.jpg"
            if not dest.exists():
                img.save(dest, format="JPEG", quality=rc["jpeg_quality"])
            yield n, img, dest
    finally:
        doc.close()


def crop(img: Image.Image, bbox: tuple[float, float, float, float], pad: float = 0.012) -> Image.Image:
    """Crop with a normalized (x1, y1, x2, y2) box, padded slightly so edge text isn't cut."""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    box = (max(0, int((x1 - pad) * w)), max(0, int((y1 - pad) * h)),
           min(w, int((x2 + pad) * w)), min(h, int((y2 + pad) * h)))
    return img.crop(box)
