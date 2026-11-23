"""Load each dataset's raw corpus and parse it into Document trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .parser import parse_article_level, parse_full_text
from .tree import Document


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def iter_documents(name: str, ds_cfg: dict, raw: Path) -> Iterator[Document]:
    corpus = raw / ds_cfg["corpus"]
    if ds_cfg["format"] == "article_level":
        # ALQAC articles start with their title line; Zalo articles have none.
        has_titles = ds_cfg.get("article_titles", True)
        for doc in _read(corpus):
            yield parse_article_level(doc["id"], doc["articles"], has_titles=has_titles)
    elif ds_cfg["format"] == "full_text":
        titles = vilegalexperts_titles(raw / ds_cfg["test"][0]) if name == "vilegalexperts" else {}
        for p in sorted(corpus.glob("*.json")):
            d = _read(p)
            yield parse_full_text(p.stem, d["passage"], title=titles.get(p.stem, d.get("name", p.stem)))
    else:
        raise ValueError(f"unknown corpus format {ds_cfg['format']!r}")


def vilegalexperts_titles(test_path: Path) -> dict[str, str]:
    """Readable document names (e.g. 'luật trồng trọt 2018') from the test-set metadata."""
    out = {}
    for q in _read(test_path).values():
        for c in q["contexts"].values():
            out[Path(c["content"]).stem] = c["document"]
    return out


def save_documents(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")


def load_documents(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document.from_dict(json.loads(line)) for line in f if line.strip()]
