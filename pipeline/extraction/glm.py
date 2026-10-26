"""GLM-4.6V (hosted, OpenAI-compatible API): reads charts and infographics, and re-reads tables
that DeepSeek-OCR got wrong."""
import re

from PIL import Image

from ..config import cfg, env_key
from ..nim_client import image_data_url, post
from .deepseek_ocr import upscale

THINK = re.compile(r"<think>.*?</think>", re.S)
BOX = re.compile(r"<\|(?:begin|end)_of_box\|>")
FENCE = re.compile(r"^```[a-z]*\n|\n?```$")


def read(img: Image.Image, prompt: str) -> str:
    c = cfg()["nims"]["glm"]
    payload = {
        "model": c["model"],
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_data_url(upscale(img, c["min_side"]))}},
            {"type": "text", "text": prompt}]}],
        "temperature": c["temperature"], "max_tokens": c["max_tokens"],
        **(c.get("extra_body") or {}),
    }
    res = post("glm", f"{c['url']}/chat/completions", payload, api_key=env_key(c["api_key_env"]))
    choice = res["choices"][0]
    if choice.get("finish_reason") == "length":
        return ""
    return clean(choice["message"].get("content") or "")


def clean(text: str) -> str:
    """GLM-V wraps final answers in box tokens and sometimes in a code fence."""
    text = BOX.sub("", THINK.sub("", text)).strip()
    return FENCE.sub("", text).strip()
