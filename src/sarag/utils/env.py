"""Load secrets from .env (project root) without ever logging their values."""

from __future__ import annotations

import os

from .config import PROJECT_ROOT


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def require_env(name: str) -> str:
    load_env()
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing environment variable {name}; set it in .env (see .env.example).")
    return val
