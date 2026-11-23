"""Document tree: Document > Article (with chapter / section metadata) > Clause > Point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Point:
    label: str                      # "a", "b", ..., "đ"
    text: str


@dataclass
class Clause:
    number: int
    text: str                       # clause text before its first point (plus continuation lines)
    points: list[Point] = field(default_factory=list)


@dataclass
class Article:
    number: str                     # "13", "13a"; "" for a fallback pseudo-article
    title: str = ""
    lead_in: str = ""               # text between the title and clause 1 (or the first point)
    clauses: list[Clause] = field(default_factory=list)
    points: list[Point] = field(default_factory=list)   # points directly under the article (no clauses)
    chapter: str = ""               # Roman numeral, e.g. "II"
    section: str = ""               # Mục number, e.g. "1"
    part: int = 0                   # 0 = main text; 1, 2, ... = attached regulation that restarts at Điều 1
    rejected_lines: int = 0         # numbered lines that broke the expected sequence (kept as text)

    def clause(self, number: int) -> Clause | None:
        return next((c for c in self.clauses if c.number == number), None)

    def full_text(self) -> str:
        parts = [self.lead_in] if self.lead_in else []
        parts += [f"{p.label}) {p.text}" for p in self.points]
        for c in self.clauses:
            parts.append(f"{c.number}. {c.text}".rstrip())
            parts += [f"{p.label}) {p.text}" for p in c.points]
        return "\n".join(parts)


@dataclass
class Document:
    doc_id: str
    title: str
    articles: list[Article] = field(default_factory=list)
    chapters: dict[str, str] = field(default_factory=dict)   # numeral -> chapter title
    preamble: str = ""
    fallback: bool = False          # no article structure found; one pseudo-article holds the text
    source_format: str = ""         # article_level | full_text

    # A part 0 with fewer articles than this is treated as a short issuing decision.
    ISSUING_DECISION_MAX_ARTICLES = 4

    def main_part(self) -> int:
        """The part holding the main regulation.

        Part 0 unless it is a short issuing decision ('Ban hành kèm theo Quyết định này Quy định ...'),
        in which case the attached part with the most text. Annexed sample forms / charters that
        also restart at Điều 1 therefore never displace a substantial part 0.
        """
        if sum(a.part == 0 for a in self.articles) > self.ISSUING_DECISION_MAX_ARTICLES:
            return 0
        size: dict[int, int] = {}
        for a in self.articles:
            size[a.part] = size.get(a.part, 0) + len(a.full_text())
        return max(size, key=size.get) if size else 0

    def article(self, number: str, part: int | None = None) -> Article | None:
        """Article by number; when several parts reuse the number, the main part wins."""
        hits = [a for a in self.articles if a.number == number and (part is None or a.part == part)]
        if len(hits) <= 1:
            return hits[0] if hits else None
        main = self.main_part()
        return next((a for a in hits if a.part == main), hits[0])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        arts = []
        for a in d["articles"]:
            clauses = [Clause(c["number"], c["text"], [Point(**p) for p in c["points"]]) for c in a["clauses"]]
            arts.append(Article(**{**a, "clauses": clauses, "points": [Point(**p) for p in a["points"]]}))
        return cls(**{**d, "articles": arts})
