"""Disk cache keyed by a stable hash (used for judge responses, embeddings, retrieval runs)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class JsonCache:
    """One JSON file per key under `root/namespace/`."""

    def __init__(self, root: str | Path, namespace: str):
        self.dir = Path(root) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        p = self._path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def set(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(key))
