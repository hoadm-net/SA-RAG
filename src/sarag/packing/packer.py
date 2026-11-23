"""Budget-aware packing (C4): choose units under a token budget and render them as a grouped context.

Context layout: one block per article, blocks ordered by their best score, units inside a block in
document order. Structured blocks open with `[<document>, Điều X. <title>]` and, with C3 on, the
article lead-in; fixed windows (B1) open with `[<document>]` only. The budget is checked by counting
tokens of the rendered context, so headers and lead-ins are always paid for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..expansion.structural import Candidate
from ..indexing.store import ArticleKey, UnitStore

TokenCounter = Callable[[str], int]


@dataclass
class Packed:
    text: str
    units: list[int]                                  # unit indices in rendered order
    articles: list[ArticleKey]                        # article keys in rendered order (citable)
    n_tokens: int
    truncated_unit: int | None = None
    blocks: list[dict] = field(default_factory=list)  # [{"doc", "article", "label", "units"}] for citation checks


class Packer:
    def __init__(self, store: UnitStore, count: TokenCounter, lead_in: bool = True,
                 truncate: Callable[[str, int], str] | None = None, min_tail: int = 32):
        self.store = store
        self.count = count
        self.lead_in = lead_in
        self.truncate = truncate          # (text, max_tokens) -> prefix; needed for topk_truncate
        self.min_tail = min_tail          # topk_truncate: do not add a cut unit shorter than this
        self._ntok: dict[int, int] = {}

    # ------------------------------------------------------------------ rendering

    def _block_key(self, i: int) -> tuple:
        u = self.store.units[i]
        if u["kind"] == "fixed":
            return ("fixed", u["doc_id"], u["unit_id"])           # windows are their own blocks
        return ("art", u["doc_id"], u["part"], u["articles"][0])

    def _header(self, key: tuple) -> str:
        if key[0] == "fixed":
            return f"[{self.store.doc_label(key[1])}]"
        _, doc_id, part, number = key
        label = self.store.doc_label(doc_id)
        art = self.store.article((doc_id, part, number))
        if not number:                                              # fallback unit: no article
            return f"[{label}]"
        title = f"Điều {number}. {art.title}".rstrip(". ") if art and art.title else f"Điều {number}"
        head = f"[{label}, {title}]"
        if self.lead_in and art and art.lead_in and art.clauses:
            head += "\n" + art.lead_in
        return head

    def _body(self, i: int, override: dict[int, str]) -> str:
        if i in override:
            return override[i]
        u = self.store.units[i]
        if u["kind"] == "article" and not u["breadcrumb"] and u["body"].startswith("Điều "):
            # B2 / no-breadcrumb article units carry their own title line; the header already has it.
            return u["body"].split("\n", 1)[1] if "\n" in u["body"] else ""
        return u["body"]

    def render(self, chosen: list[int], scores: dict[int, float], override: dict[int, str] | None = None) -> tuple[str, list[dict]]:
        override = override or {}
        blocks: dict[tuple, list[int]] = {}
        for i in chosen:
            blocks.setdefault(self._block_key(i), []).append(i)
        ordered = sorted(blocks.items(), key=lambda kv: -max(scores[i] for i in kv[1]))
        parts, meta = [], []
        for key, members in ordered:
            members.sort(key=lambda i: (self.store.units[i]["order"], self.store.units[i]["unit_id"]))
            body = "\n".join(b for b in (self._body(i, override) for i in members) if b)
            parts.append(self._header(key) + ("\n" + body if body else ""))
            u0 = self.store.units[members[0]]
            meta.append({"doc": u0["doc_id"], "part": u0["part"],
                         "articles": sorted({a for i in members for a in self.store.units[i]["articles"]}),
                         "label": self.store.doc_label(u0["doc_id"]), "units": members})
        return "\n\n".join(parts), meta

    # ------------------------------------------------------------------ selection

    def ntok(self, i: int) -> int:
        if i not in self._ntok:
            self._ntok[i] = max(self.count(self._body(i, {})), 1)
        return self._ntok[i]

    def pack(self, cands: list[Candidate], budget: int, strategy: str = "score_per_token",
             length_exponent: float = 1.0, max_candidates: int = 60) -> Packed:
        cands = sorted(cands, key=lambda c: -c.score)[:max_candidates]
        scores = {c.unit: c.score for c in cands}
        if strategy == "score_per_token":
            order = sorted(cands, key=lambda c: -c.score / self.ntok(c.unit) ** length_exponent)
            return self._greedy([c.unit for c in order], scores, budget)
        if strategy == "topk_truncate":
            return self._topk_truncate([c.unit for c in cands], scores, budget)
        raise ValueError(f"unknown packing strategy {strategy!r}")

    def _finish(self, chosen: list[int], scores: dict[int, float], override=None, truncated=None) -> Packed:
        text, meta = self.render(chosen, scores, override)
        arts = list(dict.fromkeys((b["doc"], b["part"], a) for b in meta for a in b["articles"]))
        return Packed(text, [i for b in meta for i in b["units"]], arts, self.count(text) if text else 0,
                      truncated, meta)

    def _greedy(self, order: list[int], scores: dict[int, float], budget: int) -> Packed:
        """Add units in priority order, skipping any that would overflow; stop when nothing fits."""
        chosen: list[int] = []
        for i in order:
            if self.ntok(i) > budget:
                continue
            text, _ = self.render(chosen + [i], scores)
            if self.count(text) <= budget:
                chosen.append(i)
        return self._finish(chosen, scores)

    def _topk_truncate(self, ranked: list[int], scores: dict[int, float], budget: int) -> Packed:
        """Baseline: add units in rank order; the first unit that does not fit is cut to fill the rest."""
        chosen: list[int] = []
        for i in ranked:
            text, _ = self.render(chosen + [i], scores)
            if self.count(text) <= budget:
                chosen.append(i)
                continue
            used = self.count(self.render(chosen, scores)[0]) if chosen else 0
            room = budget - used - 24                       # header + separators of the new block
            if self.truncate is not None and room >= self.min_tail:
                cut = self.truncate(self._body(i, {}), room)
                for _ in range(4):                          # shrink until the rendered context fits
                    text, _ = self.render(chosen + [i], scores, {i: cut})
                    if self.count(text) <= budget:
                        return self._finish(chosen + [i], scores, {i: cut}, truncated=i)
                    cut = self.truncate(cut, max(self.count(cut) - (self.count(text) - budget) - 8, 0))
            break
        return self._finish(chosen, scores)
