"""Answer and citation parsing; citation verification against the document tree and the packed context (C4)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from ..parsing.tree import Document

CITATION_RE = re.compile(
    r"\[\s*(?P<doc>[^\[\]]+?)\s*,\s*Điều\s+(?P<art>\d+[a-zđ]?)"
    r"(?:\s*,\s*[Kk]hoản\s+(?P<cl>\d+))?(?:\s*,\s*[Đđ]iểm\s+(?P<pt>[a-zđ]))?\s*\]")
ANSWER_RE = re.compile(r"Đáp án\s*[:：]\s*(?P<ans>.+)", re.I)


@dataclass
class Citation:
    raw: str
    doc_label: str
    article: str
    clause: int | None = None
    point: str | None = None
    doc_id: str | None = None
    status: str = ""          # valid | not_in_context | not_in_tree | unknown_document

    def to_dict(self) -> dict:
        return asdict(self)


def parse_citations(text: str) -> list[Citation]:
    return [Citation(m.group(0), m.group("doc").strip(), m.group("art"),
                     int(m.group("cl")) if m.group("cl") else None, m.group("pt")) for m in CITATION_RE.finditer(text)]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[^\w/]+", " ", s).strip()


def verify(citations: list[Citation], blocks: list[dict], docs: dict[str, Document]) -> list[Citation]:
    """Resolve each citation's document label against the packed blocks, then check that the article
    (and clause, if given) exists in the document tree and that the article is in the packed context."""
    labels: dict[str, str] = {}
    for b in blocks:
        labels.setdefault(_norm(b["label"]), b["doc"])
    in_context = {(b["doc"], b["part"], a) for b in blocks for a in b["articles"]}
    for c in citations:
        n = _norm(c.doc_label)
        doc_id = labels.get(n)
        if doc_id is None:                                   # shortened or extended label
            hits = {d for lab, d in labels.items() if lab and (lab in n or n in lab)}
            doc_id = hits.pop() if len(hits) == 1 else None
        c.doc_id = doc_id
        if doc_id is None or doc_id not in docs:
            c.status = "unknown_document"
            continue
        art = docs[doc_id].article(c.article)
        if art is None or (c.clause is not None and art.clause(c.clause) is None):
            c.status = "not_in_tree"
        elif not any((doc_id, p, c.article) in in_context for p in {art.part, 0}):
            c.status = "not_in_context"
        else:
            c.status = "valid"
    return citations


def strip_invalid(text: str, citations: list[Citation]) -> str:
    for c in citations:
        if c.status != "valid":
            text = text.replace(c.raw, "")
    return re.sub(r"[ \t]{2,}", " ", text)


def parse_answer(text: str, qtype: str) -> str:
    """Normalised short answer: 'Đúng' / 'Sai', a choice letter, or free text ('' if not found)."""
    m = ANSWER_RE.search(text)
    head = m.group("ans").strip() if m else text.strip().split("\n", 1)[0]
    if qtype == "Đúng/Sai":
        h = _norm(head)
        if h.startswith("đúng"):
            return "Đúng"
        if h.startswith("sai") or h.startswith("không đúng"):
            return "Sai"
        return ""
    if qtype == "Trắc nghiệm":
        mm = re.match(r"\s*\(?([ABCD])\b", head)
        return mm.group(1) if mm else ""
    return CITATION_RE.sub("", head).strip().rstrip(".").strip()
