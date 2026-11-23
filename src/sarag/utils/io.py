"""JSONL I/O with resume support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_ids(path: str | Path, key: str = "question_id") -> set:
    """IDs already present in a partial output file (for resuming generation)."""
    path = Path(path)
    if not path.exists():
        return set()
    return {r[key] for r in read_jsonl(path) if key in r}
