"""Loads config.yaml and .env once; every module reads settings from here."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def cfg() -> dict:
    load_dotenv(ROOT / ".env", override=True)  # .env wins over older keys set in the Windows environment
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    p = ROOT / cfg()["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def nvidia_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is not set (add it to .env or your environment).")
    return key


def env_key(name: str) -> str:
    cfg()  # loads .env
    key = os.environ.get(name)
    if not key:
        raise RuntimeError(f"{name} is not set (add it to .env or your environment).")
    return key
