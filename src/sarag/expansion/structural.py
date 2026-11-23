"""Structural expansion (C3): one-hop cross-reference resolution within the same document.

The parent lead-in (article title + introductory sentence) is rendered by the packer for every
article that contributes a unit; this module adds the units that retrieved units refer to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..indexing.store import UnitStore

_LIST_SEP = r"\s*(?:,|và|hoặc)\s*"
REF_RE = re.compile(
    r"(?:(?:các\s+)?điểm\s+(?P<pts>[a-zđ](?:" + _LIST_SEP + r"(?:điểm\s+)?[a-zđ])*)\s+)?"
    r"(?:(?:các\s+)?khoản\s+(?P<cls>\d+(?:" + _LIST_SEP + r"(?:khoản\s+)?\d+)*)\s+(?:của\s+)?)?"
    r"Điều\s+(?P<art>này|\d+[a-zđ]?(?:" + _LIST_SEP + r"(?:Điều\s+)?\d+[a-zđ]?)*)",
    re.I,
)
_DOC_TYPES = (r"Luật|Bộ luật|Nghị định|Thông tư(?: liên tịch)?|Quyết định|Pháp lệnh|Nghị quyết|Hiến pháp|"
              r"Quy định|Quy chế|Điều lệ|Quy chuẩn|Chỉ thị|Văn bản")
_THIS_DOC_RE = re.compile(r"^\s*(?:của\s+)?(?:" + _DOC_TYPES + r")?\s*này\b", re.I)
_OTHER_DOC_RE = re.compile(r"^\s*(?:của\s+|tại\s+)?(?:" + _DOC_TYPES + r")\b", re.I)


@dataclass(frozen=True)
class Reference:
    article: str              # article number, or "" for "Điều này" (the source article)
    clauses: tuple[int, ...]  # empty = whole article


def find_references(text: str) -> list[Reference]:
    """References in `text` that point into the same document.

    'Điều này' and '... của Luật/Nghị định/... này' are internal; a reference followed by another
    document's name ('Điều 53 Luật Dự trữ quốc gia', 'khoản 3 Điều 30 Nghị định số 29/2012/NĐ-CP')
    is external and skipped. A bare reference with no document named is taken as internal.
    """
    refs: list[Reference] = []
    for m in REF_RE.finditer(text):
        clauses = tuple(int(x) for x in re.findall(r"\d+", m.group("cls") or ""))
        art = m.group("art")
        if art.lower() == "này":
            refs.append(Reference("", clauses))
            continue
        tail = text[m.end():m.end() + 80]
        if not _THIS_DOC_RE.match(tail) and _OTHER_DOC_RE.match(tail):
            continue
        numbers = re.findall(r"\d+[a-zđ]?", art)
        for n in numbers:
            refs.append(Reference(n, clauses if len(numbers) == 1 else ()))
    return list(dict.fromkeys(refs))


@dataclass
class Candidate:
    unit: int
    score: float
    source: str = "retrieved"     # retrieved | xref


def expand(cands: list[Candidate], store: UnitStore, decay: float = 0.5, max_targets: int = 8) -> list[Candidate]:
    """Add the units referenced by each retrieved unit (one hop), scored `decay` x the source score.
    A unit reached several ways keeps its best score; retrieved units are never downgraded."""
    best: dict[int, Candidate] = {c.unit: c for c in cands}
    for c in cands:
        u = store.units[c.unit]
        if u["kind"] == "fixed" or u["clause"] is None and u["kind"] == "fallback":
            continue
        doc_id, part, art = u["doc_id"], u["part"], u["articles"][0]
        targets: list[int] = []
        for ref in find_references(u["body"]):
            key = (doc_id, part, art) if not ref.article else store.resolve_article(doc_id, part, ref.article)
            if key is None:
                continue
            if ref.clauses:
                for cl in ref.clauses:
                    targets += store.by_clause.get((*key, cl), [])
            else:
                targets += store.by_article.get(key, [])
        for t in list(dict.fromkeys(targets))[:max_targets]:
            if t == c.unit:
                continue
            s = c.score * decay
            if t not in best or best[t].score < s and best[t].source == "xref":
                best[t] = Candidate(t, s, "xref")
    return sorted(best.values(), key=lambda x: -x.score)
