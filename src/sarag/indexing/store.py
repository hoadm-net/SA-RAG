"""Units of one index variant plus the document trees, with structural lookups (used by C3 and C4)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..parsing.loaders import load_documents
from ..parsing.tree import Article, Document
from .units import display_title

ArticleKey = tuple[str, int, str]


class UnitStore:
    def __init__(self, units: list[dict], docs: list[Document]):
        self.units = units
        self.docs = {d.doc_id: d for d in docs}
        self.by_id = {u["unit_id"]: i for i, u in enumerate(units)}
        self.by_article: dict[ArticleKey, list[int]] = defaultdict(list)
        self.by_clause: dict[tuple[str, int, str, int], list[int]] = defaultdict(list)
        for i, u in enumerate(units):
            for a in u["articles"]:
                self.by_article[(u["doc_id"], u["part"], a)].append(i)
            if u["clause"] is not None:
                self.by_clause[(u["doc_id"], u["part"], u["articles"][0], u["clause"])].append(i)
        self._titles: dict[str, str] = {}

    @classmethod
    def load(cls, processed_dir: Path, variant: str) -> "UnitStore":
        with open(Path(processed_dir) / "index" / variant / "units.jsonl", encoding="utf-8") as f:
            units = [json.loads(line) for line in f]
        return cls(units, load_documents(Path(processed_dir) / "trees.jsonl"))

    def article_key(self, i: int) -> ArticleKey:
        u = self.units[i]
        return (u["doc_id"], u["part"], u["articles"][0])

    def article(self, key: ArticleKey) -> Article | None:
        doc = self.docs.get(key[0])
        return doc.article(key[2], part=key[1]) if doc else None

    def doc_label(self, doc_id: str) -> str:
        if doc_id not in self._titles:
            doc = self.docs.get(doc_id)
            self._titles[doc_id] = display_title(doc) if doc else doc_id
        return self._titles[doc_id]

    def resolve_article(self, doc_id: str, part: int, number: str) -> ArticleKey | None:
        """Key of article `number` in the same part, else in the document's main part."""
        if (doc_id, part, number) in self.by_article:
            return (doc_id, part, number)
        doc = self.docs.get(doc_id)
        if doc is not None:
            a = doc.article(number)
            if a is not None and (doc_id, a.part, number) in self.by_article:
                return (doc_id, a.part, number)
        return None
