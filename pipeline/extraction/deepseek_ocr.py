"""DeepSeek-OCR, self-hosted on vLLM: dense transcription of table, chart and infographic crops."""
import re

from PIL import Image

from ..config import cfg
from ..nim_client import image_data_url, post

# Prompts from DeepSeek-OCR's README; vLLM's chat template adds the <image> token. The README's markdown prompt
# starts with <|grounding|>, but on this manual that variant wrote "AB- 1234- 000" for AB-1234-000 on
# large crops and sometimes stopped after a heading (35 chars vs 2,501 on page 6), so it is left out.
PROMPTS = {
    "markdown": "Convert the document to markdown.",
    "figure": "Parse the figure.",
    "free": "Free OCR.",
}
GROUNDING = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.S)
SPECIAL = re.compile(r"<[|｜][^<>]{1,40}[|｜]>")


def upscale(img: Image.Image, min_side: int) -> Image.Image:
    """Small crops are enlarged so thin text survives the model's own resize."""
    side = max(img.size)
    if side >= min_side:
        return img
    f = min_side / side
    return img.resize((int(img.width * f), int(img.height * f)), Image.LANCZOS)


def read(img: Image.Image, mode: str = "markdown") -> str:
    """Returns the transcription, or "" if the model ran out of tokens (a sign it was looping)."""
    c = cfg()["nims"]["deepseek_ocr"]
    payload = {
        "model": c["model"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(upscale(img, c["min_side"]))}},
            {"type": "text", "text": PROMPTS[mode]}]}],
        "temperature": 0.0, "max_tokens": c["max_tokens"],
        # Grounding tags are special tokens; keep them so they can be stripped cleanly below.
        "skip_special_tokens": False,
        # NGramPerReqLogitsProcessor stops repetition loops; <td> and </td> are exempt so tables survive.
        "vllm_xargs": {"ngram_size": 30, "window_size": 90, "whitelist_token_ids": [128821, 128822]},
    }
    res = post("deepseek_ocr", f"{c['url']}/v1/chat/completions", payload)
    choice = res["choices"][0]
    if choice.get("finish_reason") == "length":
        return ""
    return clean(choice["message"].get("content") or "")


def clean(text: str) -> str:
    """Drops grounding tags (<|ref|>label<|/ref|><|det|>[[box]]<|/det|>, the content follows them)
    and leftover special tokens such as end-of-sentence."""
    text = SPECIAL.sub("", GROUNDING.sub("", text))
    return re.sub(r"\n{3,}", "\n\n", text).strip()
