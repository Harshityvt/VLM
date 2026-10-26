"""'NeMo Guardrails (Optional)': input rails before Query Processing, output rails before the response.

The rails model comes from llm.fast_model in config.yaml (the model line in config.yml is only a default),
and falls back through llm.fallback_models if NVIDIA stops serving it.
"""
import threading
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig

from ...config import cfg
from ..llm import candidates, is_gone_error, mark_gone

REFUSAL = "I can't help with that request."
_rails: dict[str, LLMRails] = {}
_lock = threading.Lock()


def _rails_for(model: str) -> LLMRails:
    with _lock:
        if model not in _rails:
            conf = RailsConfig.from_path(str(Path(__file__).parent))
            conf.models[0].model = model
            _rails[model] = LLMRails(conf)
        return _rails[model]


def _run(messages: list[dict], rail: str) -> bool:
    """True if the message passes the rail."""
    preferred = cfg()["llm"].get("fast_model") or cfg()["llm"]["model"]
    last: Exception | None = None
    for model in candidates(preferred):
        try:
            res = _rails_for(model).generate(messages=messages,
                                             options={"rails": [rail], "log": {"activated_rails": True}})
        except Exception as e:  # nemoguardrails wraps provider errors in its own exception types
            if not is_gone_error(e):
                raise
            mark_gone(model, e)
            last = e
            continue
        return not (res.log and any(r.stop for r in res.log.activated_rails))
    raise RuntimeError(f"No hosted LLM is available for guardrails: {last}")


def check_input(question: str) -> bool:
    return _run([{"role": "user", "content": question}], "input")


def check_output(question: str, answer: str) -> bool:
    return _run([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], "output")
