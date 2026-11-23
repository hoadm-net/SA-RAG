"""Regular expressions and text normalisation for Vietnamese legal documents."""

from __future__ import annotations

import re

# Vietnamese point labels in order (no f, j, w, z; đ follows d).
POINT_LABELS = list("abcdđeghiklmnopqrstuvxy")
NEXT_POINT = {a: b for a, b in zip(POINT_LABELS, POINT_LABELS[1:])}

CHAPTER_RE = re.compile(r"^Chương\s+([IVXLC]+|\d+)\b\.?\s*[:\-–]?\s*(.*)$")
SECTION_RE = re.compile(r"^Mục\s+(\d+)\b\.?\s*[:\-–]?\s*(.*)$")
# Trailer tags used by some restructured documents: "Chương: II", "Mục: 3".
TAG_RE = re.compile(r"^(Chương|Mục)\s*:\s*([IVXLC]+|\d+)\s*$")
ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*[\.:]\s*(?P<dash>[-–—]\s*)?(?P<rest>.*)$")
CLAUSE_RE = re.compile(r"^(\d{1,3})\.(?!\d)\s*(.*)$")
POINT_RE = re.compile(r"^([a-zđ])\)\s*(.*)$")

# Signature blocks, annexes and enactment formulas that follow the last article.
FOOTER_RE = re.compile(
    r"^(Nơi nhận\s*:|TM\.|T/M|KT\.|K\.T\.|Q\.\s|CHỦ TỊCH|PHÓ CHỦ TỊCH|THỦ TƯỚNG|PHÓ THỦ TƯỚNG|BỘ TRƯỞNG|"
    r"THỨ TRƯỞNG|PHỤ LỤC|MẪU SỐ|"
    r"(Bộ luật|Luật|Pháp lệnh|Nghị quyết) này (đã )?được .{0,120}thông qua)"
)
# Heading of a regulation issued together with (attached to) a decision.
ATTACHED_RE = re.compile(r"(ban hành|kèm theo)\s.{0,40}(Quyết định|Thông tư|Nghị định|Nghị quyết)", re.I)
MAX_TITLE_CHARS = 300

_SPACES = re.compile(r"[ \t ]+")


def normalize_text(text: str) -> str:
    """Join soft line wraps and tidy whitespace.

    Scraped full texts wrap long lines as "\\r\\n\\n" while real paragraph breaks are "\\n" or "\\n\\n".
    """
    text = text.replace("\r\n\n", " ").replace("\r\n", " ").replace("\r", " ")
    lines = (_SPACES.sub(" ", ln).strip() for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def is_heading_title(s: str) -> bool:
    """Chapter / section titles are upper case (digits and punctuation allowed)."""
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)
