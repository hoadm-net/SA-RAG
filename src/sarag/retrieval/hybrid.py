"""Hybrid retrieval (BM25 + bge-m3 fused with RRF) and clause-to-article aggregation (C2)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..indexing.bm25 import BM25Index
from ..indexing.dense import DenseIndex

ArticleKey = tuple[str, int, str]


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for r, doc in enumerate(ranking, start=1):
            scores[doc] += w / (k + r)
    return dict(scores)


def aggregate(unit_scores: list[tuple[int, float]], unit_articles: list[list[ArticleKey]],
              method: str = "max") -> list[tuple[ArticleKey, float]]:
    """Article ranking from scored units (sorted by decreasing score).

    max: article score = best unit score. sum_top2: sum of the two best unit scores.
    none: articles in order of their first unit (no score aggregation); scores are the unit scores.
    A unit covering several articles (B1 windows) contributes to each of them.
    """
    per_article: dict[ArticleKey, list[float]] = defaultdict(list)
    for u, s in unit_scores:
        for a in unit_articles[u]:
            per_article[a].append(s)
    if method in ("max", "none"):
        agg = {a: max(v) for a, v in per_article.items()}
    elif method == "sum_top2":
        agg = {a: sum(sorted(v, reverse=True)[:2]) for a, v in per_article.items()}
    else:
        raise ValueError(f"unknown aggregation {method!r}")
    if method == "none":
        order = list(dict.fromkeys(a for u, _ in unit_scores for a in unit_articles[u]))
        return [(a, agg[a]) for a in order]
    return sorted(agg.items(), key=lambda x: -x[1])


@dataclass
class RetrievalResult:
    units: list[tuple[int, float]]                 # (unit index, fused score), best first
    articles: list[tuple[ArticleKey, float]]       # aggregated, best first


class HybridRetriever:
    """Loads one index variant (units + BM25 + dense vectors) built by scripts/03_build_index.py."""

    def __init__(self, index_dir: Path, use_bm25: bool = True, use_dense: bool = True, encoder=None):
        index_dir = Path(index_dir)
        with open(index_dir / "units.jsonl", encoding="utf-8") as f:
            self.units = [json.loads(line) for line in f]
        self.unit_articles: list[list[ArticleKey]] = [
            [(u["doc_id"], u["part"], a) for a in u["articles"]] for u in self.units]
        self.bm25 = BM25Index.load(index_dir / "bm25") if use_bm25 else None
        self.dense = DenseIndex.load(index_dir / "dense.npy") if use_dense else None
        if self.dense is not None and encoder is None:
            raise ValueError("dense retrieval needs a query encoder")
        self.encoder = encoder

    def search(self, queries: list[str], top_k_units: int = 50, rrf_k: int = 60, aggregation: str = "max",
               candidates: int = 200, bm25_weight: float = 1.0, dense_weight: float = 1.0) -> list[RetrievalResult]:
        """Each retriever returns `candidates` units; RRF fuses them; the best `top_k_units` are kept
        and aggregated to articles."""
        lists: list[np.ndarray] = []
        weights: list[float] = []
        if self.bm25 is not None:
            lists.append(self.bm25.search(queries, candidates)[0])
            weights.append(bm25_weight)
        if self.dense is not None:
            lists.append(self.dense.search(self.encoder.encode(queries), candidates)[0])
            weights.append(dense_weight)
        out = []
        for qi in range(len(queries)):
            fused = rrf([[int(x) for x in lst[qi]] for lst in lists], k=rrf_k, weights=weights)
            units = sorted(fused.items(), key=lambda x: -x[1])[:top_k_units]
            out.append(RetrievalResult(units, aggregate(units, self.unit_articles, aggregation)))
        return out
