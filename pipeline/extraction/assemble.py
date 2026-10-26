"""Runs the extraction pipeline for one page and returns chunk records ready for embedding.

Pages as images -> Page Elements -> {Tables -> Table Structure -> OCR (DeepSeek-OCR, GLM-4.6V re-read),
                                      Charts -> Graphic Elements -> OCR (GLM-4.6V),
                                      Infographics -> OCR (GLM-4.6V)}
PDF pages -> Parse -> text + metadata
"""
import logging
from pathlib import Path

from PIL import Image

from ..config import ROOT, cfg
from . import graphic_elements, ocr, page_elements, parse, table_structure
from .render import crop

log = logging.getLogger(__name__)


def page_chunks(page_no: int, img: Image.Image, img_path: Path) -> list[dict]:
    rel_img = str(img_path.relative_to(ROOT)).replace("\\", "/")
    base = {"page": page_no, "page_image": rel_img}
    chunks: list[dict] = []

    # Parse path (text and metadata). Runs first so the VLMs can be told which section a crop comes from.
    blocks = parse.parse_page(img)
    header = page_header(blocks)

    # Detection path
    regions = page_elements.detect([img])[0]
    tables = [r for r in regions if r.label == "table"]
    charts = [r for r in regions if r.label == "chart"]
    infos = [r for r in regions if r.label == "infographic"]

    if tables:
        crops = [crop(img, t.box) for t in tables]
        structures = table_structure.detect(crops)
        for t, r in zip(tables, ocr.read_tables(crops, structures, header)):
            if r.text.strip():
                chunks.append({**base, "element_type": "table", "bbox": list(t.box), "text": r.text,
                               "extractor": r.extractor})

    if charts:
        crops = [crop(img, c.box) for c in charts]
        hints = [graphic_elements.hints(e) for e in graphic_elements.detect(crops)]
        for c, r in zip(charts, ocr.read_charts(crops, hints, header)):
            if r.text.strip():
                chunks.append({**base, "element_type": "chart", "bbox": list(c.box), "text": r.text,
                               "extractor": r.extractor})

    if infos:
        crops = [crop(img, i.box) for i in infos]
        for i, r in zip(infos, ocr.read_infographics(crops, header)):
            if r.text.strip():
                chunks.append({**base, "element_type": "infographic", "bbox": list(i.box), "text": r.text,
                               "extractor": r.extractor})

    parse_name = cfg()["nims"]["parse"]["model"]
    for piece in split_text(parse.page_text(blocks)):
        chunks.append({**base, "element_type": "text", "bbox": None, "text": piece, "extractor": parse_name})

    # Prefix every chunk with the page's title lines, so a spec table or paragraph still carries the
    # product/section name that appears only once at the top of the page.
    if header:
        for c in chunks:
            if header not in c["text"]:
                c["text"] = f"[{header}]\n{c['text']}"
    return chunks


def page_header(blocks: list[parse.Block], max_chars: int = 200) -> str:
    """Titles, section headers and short standalone lines from the top of the page, in reading order."""
    parts: list[str] = []
    for b in blocks:
        t = " ".join(b.text.replace("*", "").replace("#", "").split())
        if not t or b.cls in ("Footnote", "Page-footer", "Formula"):
            continue
        if b.cls in ("Title", "Section-header") or len(t) <= 60:
            if t not in parts:
                parts.append(t)
        if sum(len(p) + 3 for p in parts) >= max_chars:
            break
    return " | ".join(parts)[:max_chars]


def split_text(text: str) -> list[str]:
    """~512-token chunks on paragraph boundaries, with overlap; headings start a new chunk when possible."""
    cc = cfg()["chunking"]
    max_c, ov = cc["max_chars"], cc["overlap_chars"]
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > max_c:  # one huge paragraph: hard split
            head, p = p[:max_c], p[max_c - ov:]
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(head)
        starts_section = p.startswith("## ") and len(cur) > max_c // 3
        if cur and (len(cur) + len(p) + 2 > max_c or starts_section):
            chunks.append(cur)
            tail = cur[-ov:] if not starts_section else ""
            cur = (tail + "\n\n" + p).strip() if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks
