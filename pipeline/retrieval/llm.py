"""'LLM' and 'LLM (optional)' boxes: hosted Nemotron models via the OpenAI-compatible API.

NVIDIA retires hosted models without notice, so every call falls back through llm.fallback_models
when a model answers 404/410 ("no longer served").
"""
import logging
import re
import threading

from openai import APIStatusError, OpenAI

from ..config import cfg, nvidia_api_key

log = logging.getLogger(__name__)
_client: OpenAI | None = None
_gone: set[str] = set()       # models that answered 404/410 during this process
_lock = threading.Lock()


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=cfg()["llm"]["base_url"], api_key=nvidia_api_key())
    return _client


def candidates(model: str) -> list[str]:
    """model first, then the configured fallbacks, skipping any already found to be gone."""
    order = [model, *cfg()["llm"].get("fallback_models", [])]
    seen, out = set(), []
    for m in order:
        if m not in seen and m not in _gone:
            seen.add(m)
            out.append(m)
    return out


def mark_gone(model: str, err: Exception) -> None:
    with _lock:
        if model not in _gone:
            _gone.add(model)
            log.warning("hosted model %s is not being served (%s); falling back", model, str(err)[:120])


def is_gone_error(err: Exception) -> bool:
    status = getattr(err, "status_code", None) or getattr(err, "status", None)
    return status in (404, 410) or bool(re.search(r"\[(404|410)\]|Not found for account|end of life", str(err)))


def chat(messages: list[dict], model: str | None = None, temperature: float | None = None,
         max_tokens: int | None = None, reasoning: bool = False) -> str:
    lc = cfg()["llm"]
    last: Exception | None = None
    for m in candidates(model or lc["model"]):
        try:
            # Nemotron 3 toggles reasoning through the chat template; "/no_think" prompts have no effect.
            resp = client().chat.completions.create(
                model=m,
                messages=messages,
                temperature=lc["temperature"] if temperature is None else temperature,
                max_tokens=max_tokens or lc["max_tokens"],
                extra_body={"chat_template_kwargs": {"enable_thinking": reasoning}},
            )
        except APIStatusError as e:
            if not is_gone_error(e):
                raise
            mark_gone(m, e)
            last = e
            continue
        text = resp.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    raise RuntimeError(f"No hosted LLM is available; tried {candidates(model or lc['model']) or 'all'}: {last}")
