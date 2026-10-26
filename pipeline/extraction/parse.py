"""'NeMo Retriever Parse': page image -> ordered text blocks with class labels ('Text and Metadata')."""
import re
from dataclasses import dataclass

from PIL import Image

from ..config import cfg
from ..nim_client import image_data_url, post

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
BLOCK = re.compile(r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>", re.S)
# Tables and pictures come from the Page Elements path instead, so they are not duplicated here.
SKIP = {"Table", "Picture", "Page-header", "Page-footer"}


@dataclass
class Block:
    cls: str
    text: str


def parse_page(img: Image.Image) -> list[Block]:
    c = cfg()["nims"]["parse"]
    payload = {
        "model": c["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_data_url(img)}}]}],
        "temperature": 0.0, "repetition_penalty": 1.1, "max_tokens": 8192,
    }
    res = post("parse", f"{c['url']}/v1/chat/completions", payload)
    msg = res["choices"][0]["message"]
    content = msg.get("content") or ""
    blocks = []
    for m in BLOCK.finditer(content):
        cls = "Formula" if m.group(6) == "Inline-formula" else m.group(6)
        text = re.sub(r"<tbc>|\\<\|unk\|\\>", "", m.group(3)).strip()
        if text and cls not in SKIP:
            blocks.append(Block(cls, text))
    if not blocks and content and not BLOCK.search(content):
        blocks.append(Block("Text", re.sub(r"<[^>]+>", "", content).strip()))
    return blocks


def page_text(blocks: list[Block]) -> str:
    out = []
    for b in blocks:
        if b.cls in ("Title", "Section-header"):
            out.append(f"## {b.text.lstrip('#').strip()}")
        elif b.cls == "List-item":
            out.append(f"- {b.text.lstrip('-• ').strip()}")
        else:
            out.append(b.text)
    return "\n\n".join(out)
