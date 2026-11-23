"""Index units built from document trees (C1) and for the baselines (B1 fixed windows, B2 articles).

Every unit records which article(s) it belongs to, so retrieval can always be scored at the article
level. Units longer than `max_tokens` are split on line boundaries so they fit the dense encoder
and can be packed under a token budget.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Iterator

from ..parsing.tree import Article, Clause, Document

TokenCounter = Callable[[str], int]

_DOC_TYPES = [  # Zalo-style document codes -> readable type
    ("ttlt", "Thông tư liên tịch"), ("tt", "Thông tư"), ("nđ", "Nghị định"), ("nd", "Nghị định"),
    ("qđ", "Quyết định"), ("nq", "Nghị quyết"), ("qh", "Luật"), ("pl", "Pháp lệnh"), ("lh", "Luật"),
]
_CODE_RE = re.compile(r"^\d+/\d{4}/([a-zđð]+)")


@dataclass
class Unit:
    unit_id: str
    doc_id: str
    kind: str                        # clause | article | fallback | fixed
    body: str                        # unit text without breadcrumb
    breadcrumb: str = ""
    articles: list[str] = field(default_factory=list)   # article numbers covered (main key for scoring)
    part: int = 0
    clause: int | None = None
    order: int = 0                   # position in the document, for keeping original order when packing
    n_tokens: int = 0                # tokens of `text` under the counter used at build time

    @property
    def text(self) -> str:
        return f"{self.breadcrumb}\n{self.body}" if self.breadcrumb else self.body

    def to_dict(self) -> dict:
        return asdict(self)


def display_title(doc: Document) -> str:
    """Readable document title for breadcrumbs ('47/2011/tt-bca' -> 'Thông tư 47/2011/TT-BCA')."""
    t = doc.title.strip()
    m = _CODE_RE.match(t.lower())
    if m and t == doc.doc_id:
        kind = next((name for code, name in _DOC_TYPES if m.group(1).startswith(code)), "Văn bản")
        return f"{kind} {t.upper()}"
    return t[:1].upper() + t[1:]


def article_label(a: Article) -> str:
    if not a.number:
        return ""
    return f"Điều {a.number}. {a.title}".rstrip(". ") if a.title else f"Điều {a.number}"


def breadcrumb(doc: Document, a: Article, clause: Clause | None = None) -> str:
    parts = [display_title(doc)]
    if article_label(a):
        parts.append(article_label(a))
    if clause is not None:
        parts.append(f"Khoản {clause.number}")
    return " > ".join(parts)


def clause_body(c: Clause) -> str:
    return "\n".join([f"{c.number}. {c.text}".rstrip()] + [f"{p.label}) {p.text}" for p in c.points])


def article_body(a: Article, with_title: bool = True) -> str:
    head = article_label(a) if with_title else ""
    return "\n".join(x for x in [head, a.full_text()] if x)


# --------------------------------------------------------------------------- unit builders

def clause_units(doc: Document, breadcrumbs: bool = True) -> Iterator[Unit]:
    """SA-RAG units: one per clause (with its points); an article without clauses is one unit.
    The article lead-in is not part of a clause unit; C3 adds it back at packing time."""
    order = 0
    for a in doc.articles:
        kind = "fallback" if doc.fallback else "clause"
        if a.clauses:
            for c in a.clauses:
                yield Unit(f"{doc.doc_id}|{a.part}|{a.number}|{c.number}", doc.doc_id, kind, clause_body(c),
                           breadcrumb(doc, a, c) if breadcrumbs else "", [a.number], a.part, c.number, order)
                order += 1
        else:
            body = a.full_text() if breadcrumbs else article_body(a)
            yield Unit(f"{doc.doc_id}|{a.part}|{a.number}|", doc.doc_id, "fallback" if doc.fallback else "article",
                       body, breadcrumb(doc, a) if breadcrumbs else "", [a.number], a.part, None, order)
            order += 1


def article_units(doc: Document, breadcrumbs: bool = False) -> Iterator[Unit]:
    """B2 baseline: one unit per article (title + full text)."""
    for order, a in enumerate(doc.articles):
        yield Unit(f"{doc.doc_id}|{a.part}|{a.number}|", doc.doc_id, "article",
                   a.full_text() if breadcrumbs else article_body(a),
                   breadcrumb(doc, a) if breadcrumbs else "", [a.number], a.part, None, order)


def fixed_units(doc: Document, count: TokenCounter, size: int = 256, overlap: int = 32,
                tokenize: Callable[[str], list[str]] | None = None) -> Iterator[Unit]:
    """B1 baseline: sliding windows of `size` tokens (stride size - overlap) over the document text,
    ignoring structure. Each window records the articles it overlaps, the one with most tokens in the
    window first (so a window's article order does not depend on where the window starts). A window
    spanning two parts keeps the articles of its majority part."""
    assert tokenize is not None, "fixed units need a tokenizer returning token strings"
    toks: list[str] = []
    owner: list[tuple[int, str]] = []          # (part, article number) per token
    for a in doc.articles:
        t = tokenize(article_body(a) + "\n")
        toks += t
        owner += [(a.part, a.number)] * len(t)
    stride = size - overlap
    for k, start in enumerate(range(0, max(len(toks) - overlap, 1), stride)):
        window = toks[start:start + size]
        share = Counter(owner[start:start + size])          # insertion order breaks ties by position
        ranked = [key for key, _ in sorted(share.items(), key=lambda x: -x[1])]
        part = ranked[0][0] if ranked else 0
        arts = [n for p, n in ranked if p == part]
        yield Unit(f"{doc.doc_id}|w{k}", doc.doc_id, "fixed", _detok(window), "", arts, part, None, k)


def _detok(pieces: list[str]) -> str:
    """Join sentencepiece tokens back into text."""
    return "".join(pieces).replace("▁", " ").strip()


# --------------------------------------------------------------------------- length control

def split_long(units: Iterable[Unit], count: TokenCounter, max_tokens: int) -> Iterator[Unit]:
    """Split units whose text exceeds `max_tokens` into consecutive line-based pieces.
    Each piece keeps the breadcrumb (marked with its piece number) and the article mapping."""
    for u in units:
        u.n_tokens = count(u.text)
        if u.n_tokens <= max_tokens:
            yield u
            continue
        head = count(u.breadcrumb) + 12 if u.breadcrumb else 0   # room for the " (phần i/n)" marker
        pieces, cur, cur_n = [], [], 0
        for line in _split_lines(u.body, count, max_tokens - head):
            n = count(line)
            if cur and cur_n + n > max_tokens - head:
                pieces.append("\n".join(cur))
                cur, cur_n = [], 0
            cur.append(line)
            cur_n += n
        if cur:
            pieces.append("\n".join(cur))
        for i, body in enumerate(pieces):
            crumb = f"{u.breadcrumb} (phần {i + 1}/{len(pieces)})" if u.breadcrumb else ""
            v = Unit(f"{u.unit_id}#{i}", u.doc_id, u.kind, body, crumb, u.articles, u.part, u.clause, u.order)
            v.n_tokens = count(v.text)
            yield v


def _split_lines(text: str, count: TokenCounter, limit: int) -> Iterator[str]:
    """Lines of `text`; a single line longer than `limit` is cut at sentence ends, then at words."""
    for line in text.split("\n"):
        if count(line) <= limit:
            yield line
            continue
        sents = re.split(r"(?<=[\.;:])\s+", line)
        buf = ""
        for s in sents:
            if count(s) > limit:                 # pathological sentence: cut by words
                words = s.split()
                step = max(limit // 2, 1)        # ~2 tokens per Vietnamese word, conservative
                for j in range(0, len(words), step):
                    if buf:
                        yield buf
                        buf = ""
                    yield " ".join(words[j:j + step])
                continue
            cand = f"{buf} {s}".strip()
            if buf and count(cand) > limit:
                yield buf
                buf = s
            else:
                buf = cand
        if buf:
            yield buf


def build_units(docs: Iterable[Document], unit: str, breadcrumbs: bool, count: TokenCounter,
                max_tokens: int, fixed_size: int = 256, fixed_overlap: int = 32,
                tokenize: Callable[[str], list[str]] | None = None) -> list[Unit]:
    out: list[Unit] = []
    for doc in docs:
        if unit == "clause":
            us = clause_units(doc, breadcrumbs)
        elif unit == "article":
            us = article_units(doc, breadcrumbs)
        elif unit == "fixed":
            us = fixed_units(doc, count, fixed_size, fixed_overlap, tokenize)
        else:
            raise ValueError(f"unknown unit type {unit!r}")
        out.extend(split_long(us, count, max_tokens) if unit != "fixed" else _count_all(us, count))
    return out


def _count_all(units: Iterable[Unit], count: TokenCounter) -> Iterator[Unit]:
    for u in units:
        u.n_tokens = count(u.text)
        yield u
