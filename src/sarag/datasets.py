"""Test questions of every dataset in one format, with gold articles as (doc_id, part, article) keys."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .parsing.tree import Document

ArticleKey = tuple[str, int, str]       # (doc_id, part, article number)

_REF_RE = re.compile(r"(?:khoản\s+(\d+)\s+)?Điều\s+(\d+[a-zđ]?)", re.I)
# A reference followed by another document's name ("Điều 5 Nghị định 124/2021/NĐ-CP") points elsewhere.
_OTHER_DOC_RE = re.compile(r"^\s*(của\s+)?(Luật|Bộ luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Hiến pháp|"
                           r"Pháp lệnh|Quy chuẩn|Điều lệ)\s*(số\s*)?([\w/\.:-]*)", re.I)


@dataclass
class Question:
    qid: str
    text: str
    gold_articles: list[ArticleKey] = field(default_factory=list)
    gold_docs: list[str] = field(default_factory=list)
    qtype: str = ""                      # Đúng/Sai | Trắc nghiệm | Tự luận | "" (retrieval only / long-form)
    choices: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    silver: bool = False                 # gold articles parsed from the reference answer


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_questions(name: str, ds_cfg: dict, raw: Path, docs: list[Document] | None = None,
                   split: str = "test") -> list[Question]:
    files = ds_cfg[split] if isinstance(ds_cfg[split], list) else [ds_cfg[split]]
    if ds_cfg["format"] == "article_level":
        out = []
        for f in files:
            for q in _read(raw / f):
                out.append(Question(
                    qid=str(q.get("question_id", q.get("id"))), text=q["text"],
                    gold_articles=list(dict.fromkeys((r["law_id"], 0, str(r["article_id"]))
                                                     for r in q["relevant_articles"])),
                    gold_docs=list(dict.fromkeys(r["law_id"] for r in q["relevant_articles"])),
                    qtype=q.get("question_type", ""), choices=q.get("choices") or {}, answer=str(q.get("answer", ""))))
        return out
    if name == "vilegalexperts":
        return _vilegalexperts(raw / files[0], docs or [], ds_cfg.get("require_released_context", True))
    raise ValueError(name)


def _vilegalexperts(path: Path, docs: list[Document], require_context: bool) -> list[Question]:
    by_id = {d.doc_id: d for d in docs}
    out = []
    for qid, q in _read(path).items():
        ctx = next(iter(q["contexts"].values()))
        doc_id = Path(ctx["content"]).stem
        doc = by_id.get(doc_id)
        if doc is None and require_context:
            continue
        out.append(Question(qid=str(qid), text=q["question"], gold_docs=[doc_id], answer=q["answer"],
                            gold_articles=silver_articles(q["answer"], doc) if doc else [], silver=True))
    return out


def silver_articles(answer: str, doc: Document) -> list[ArticleKey]:
    """Article references in an expert answer that point to the gold document and exist in its tree."""
    keys: list[ArticleKey] = []
    title = doc.title.lower()
    for m in _REF_RE.finditer(answer):
        nxt = answer[m.end():m.end() + 60]
        other = _OTHER_DOC_RE.match(nxt)
        if other and "này" not in nxt[:25]:
            code = (other.group(4) or "").lower().strip(".")
            if not code or code not in title:
                continue
        art = doc.article(m.group(2))
        if art is not None:
            keys.append((doc.doc_id, art.part, art.number))
    return list(dict.fromkeys(keys))
