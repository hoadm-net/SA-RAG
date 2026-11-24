"""Article-level retrieval metrics and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def recall_at_k(ranked: list, gold: set, k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else float("nan")


def mrr_at_k(ranked: list, gold: set, k: int = 10) -> float:
    for r, x in enumerate(ranked[:k], start=1):
        if x in gold:
            return 1.0 / r
    return 0.0


def retrieval_metrics(ranked_lists: list[list], golds: list[set], ks=(1, 3, 5, 10), mrr_k: int = 10) -> dict:
    """Means over questions with at least one gold item, plus per-question values for bootstrapping."""
    per_q = {f"R@{k}": [] for k in ks}
    per_q[f"MRR@{mrr_k}"] = []
    for ranked, gold in zip(ranked_lists, golds):
        if not gold:
            continue
        for k in ks:
            per_q[f"R@{k}"].append(recall_at_k(ranked, gold, k))
        per_q[f"MRR@{mrr_k}"].append(mrr_at_k(ranked, gold, mrr_k))
    means = {m: float(np.mean(v)) if v else float("nan") for m, v in per_q.items()}
    means["n"] = len(per_q[f"MRR@{mrr_k}"])
    return {"mean": means, "per_question": per_q}


def bootstrap_ci(values: list[float], samples: int = 1000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(samples, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def cluster_bootstrap_diff(diffs: dict[str, list[float]], samples: int = 2000, ci: float = 0.95,
                           seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference with a bootstrap CI that resamples questions (clusters): all values of a
    question (e.g. one per generator) are kept together, so pooling models does not shrink the CI."""
    keys = list(diffs)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    sums = np.array([sum(diffs[k]) for k in keys], dtype=float)
    counts = np.array([len(diffs[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(samples, len(keys)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(sums.sum() / counts.sum()), float(lo), float(hi)
