"""Parse Vietnamese legal text into Document trees (component C1).

Two entry points:
- `parse_article_level`: corpora already split into articles (ALQAC, Zalo).
- `parse_full_text`: full documents with Chương / Mục / Điều headings (ViLegalExperts).
Both share `parse_article_body`, which splits an article into lead-in, clauses and points.
"""

from __future__ import annotations

from .patterns import (ARTICLE_RE, ATTACHED_RE, CHAPTER_RE, CLAUSE_RE, FOOTER_RE, MAX_TITLE_CHARS, NEXT_POINT,
                       POINT_RE, SECTION_RE, TAG_RE, is_heading_title, normalize_text)
from .tree import Article, Clause, Document, Point


def _join(a: str, b: str) -> str:
    return f"{a}\n{b}" if a else b


def parse_article_body(lines: list[str], article: Article) -> Article:
    """Fill `article.lead_in`, `.clauses` and `.points` from body lines (title already removed).

    Clauses must be numbered 1, 2, 3, ... and points a, b, c, d, đ, e, ... within their parent.
    A numbered line that breaks the sequence (a year, an amount, a nested list) is kept as
    continuation text and counted in `article.rejected_lines`.
    """
    clause: Clause | None = None
    point: Point | None = None
    for line in lines:
        m = CLAUSE_RE.match(line)
        if m:
            n = int(m.group(1))
            expected = clause.number + 1 if clause else 1
            if n == expected:
                clause = Clause(n, m.group(2))
                article.clauses.append(clause)
                point = None
                continue
            article.rejected_lines += 1
        m = POINT_RE.match(line)
        if m:
            siblings = clause.points if clause else article.points
            expected = NEXT_POINT.get(siblings[-1].label) if siblings else "a"
            if m.group(1) == expected:
                point = Point(m.group(1), m.group(2))
                siblings.append(point)
                continue
            article.rejected_lines += 1
        # Continuation of the innermost open element.
        if point is not None:
            point.text = _join(point.text, line)
        elif clause is not None:
            clause.text = _join(clause.text, line)
        else:
            article.lead_in = _join(article.lead_in, line)
    return article


# --------------------------------------------------------------------------- article-level corpora

def parse_article_level(doc_id: str, articles: list[dict], title: str | None = None,
                        has_titles: bool = True) -> Document:
    """`articles` = [{"id", "text"}]. If `has_titles`, the first line is the article title unless it
    already looks like clause 1 (some ALQAC articles, e.g. in Hiến pháp, have no title)."""
    doc = Document(doc_id=doc_id, title=title or doc_id, source_format="article_level")
    for a in articles:
        lines = normalize_text(a["text"]).split("\n")
        art = Article(number=str(a["id"]))
        if has_titles and lines and lines[0] and not CLAUSE_RE.match(lines[0]):
            art.title, lines = lines[0], lines[1:]
        doc.articles.append(parse_article_body(lines, art))
    if doc.articles:
        doc.articles[-1] = cut_footer(doc.articles[-1])
    return doc


# --------------------------------------------------------------------------- full-text documents

def _article_number_key(num: str) -> tuple[int, str]:
    digits = "".join(ch for ch in num if ch.isdigit())
    return int(digits), num[len(digits):]


def cut_footer(a: Article) -> Article:
    """Drop the signature block / annexes / enactment formula that may trail an article."""
    lines = [ln for ln in a.full_text().split("\n") if ln]
    for k, ln in enumerate(lines):
        if (k > 0 or a.title) and FOOTER_RE.match(ln):   # never empty a title-less article
            return parse_article_body(lines[:k], Article(number=a.number, title=a.title, chapter=a.chapter,
                                                         section=a.section, part=a.part))
    return a


def _starts_attachment(body: list[str]) -> bool:
    """True if the current article ends with a signature block or an 'attached regulation' heading,
    i.e. a following 'Điều 1.' starts a new part rather than being a quotation."""
    return any(FOOTER_RE.match(ln) or ATTACHED_RE.search(ln) for ln in body)


def parse_full_text(doc_id: str, text: str, title: str = "") -> Document:
    """Parse a full legal document.

    Articles must appear in increasing order, which filters out article headings quoted inside
    amendments. The exception is an attached regulation: after a signature block or an
    'ban hành kèm theo ...' heading, numbering may restart at Điều 1 and a new `part` begins.
    The signature block / annexes at the end of each part are cut. Documents without any article
    become a single fallback pseudo-article.
    """
    lines = normalize_text(text).split("\n")
    doc = Document(doc_id=doc_id, title=title or doc_id, source_format="full_text")

    chapter = section = ""
    part = 0
    cur: Article | None = None
    body: list[str] = []
    preamble: list[str] = []
    last_key: tuple[int, str] | None = None
    heading_seen = False                       # chapter heading since the current article began

    def close() -> None:
        if cur is not None:
            doc.articles.append(parse_article_body(body, cur))

    i = 0
    while i < len(lines):
        line = lines[i]
        tag = TAG_RE.match(line)
        if tag:                                    # trailer tag belongs to the article just read
            if cur is not None:
                if tag.group(1) == "Chương":
                    cur.chapter = tag.group(2)
                else:
                    cur.section = tag.group(2)
            i += 1
            continue
        m = CHAPTER_RE.match(line)
        if m and (not m.group(2) or is_heading_title(m.group(2))):
            chapter, section, heading_seen = m.group(1), "", True
            name = m.group(2)
            if not name and i + 1 < len(lines) and is_heading_title(lines[i + 1]):
                name, i = lines[i + 1], i + 1
            doc.chapters.setdefault(chapter if part == 0 else f"{part}:{chapter}", name)
            i += 1
            continue
        m = SECTION_RE.match(line)
        if m and (not m.group(2) or is_heading_title(m.group(2))):
            section = m.group(1)
            if not m.group(2) and i + 1 < len(lines) and is_heading_title(lines[i + 1]):
                i += 1
            i += 1
            continue
        m = ARTICLE_RE.match(line)
        if m:
            key = _article_number_key(m.group(1))
            restart = (last_key is not None and key == (1, "") and cur is not None and _starts_attachment(body))
            if last_key is None or key > last_key or restart:
                close()
                if restart:
                    doc.articles[-1] = cut_footer(doc.articles[-1])
                    part += 1
                    if not heading_seen:           # chapters of the decision do not carry over
                        chapter = section = ""
                last_key, heading_seen = key, False
                cur, body = Article(number=m.group(1), chapter=chapter, section=section, part=part), []
                rest = m.group("rest").strip()
                if m.group("dash") or len(rest) > MAX_TITLE_CHARS:
                    body.append(rest)              # "Điều 1. - <content>": no title
                else:
                    cur.title = rest
                i += 1
                continue
        (body if cur is not None else preamble).append(line)
        i += 1
    close()

    if doc.articles:
        doc.articles[-1] = cut_footer(doc.articles[-1])
    doc.preamble = "\n".join(preamble)
    if not doc.articles:
        doc.fallback = True
        doc.preamble = ""
        doc.articles = [parse_article_body(preamble, Article(number=""))]
    return doc
