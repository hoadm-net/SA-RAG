"""QA and citation metrics for one prediction file."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

from ..datasets import Question
from .retrieval import bootstrap_ci


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s).lower()
    return re.findall(r"\w+", s)


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def citation_scores(pred: dict, q: Question, verified: bool) -> dict:
    """Precision / recall of cited articles against gold articles (document + article number).
    With verification on, only citations that passed verification count (invalid ones were removed)."""
    cites = [c for c in pred["citations"] if c["doc_id"] and (c["status"] == "valid" or not verified)]
    cited = {(c["doc_id"], c["article"]) for c in cites}
    gold = {(d, a) for d, _, a in q.gold_articles}
    out = {"n_citations": len(pred["citations"]), "n_kept": len(cites),
           "in_tree": sum(c["status"] in ("valid", "not_in_context") for c in pred["citations"])}
    if gold:
        out["cit_recall"] = len(cited & gold) / len(gold)
        out["cit_precision"] = len(cited & gold) / len(cited) if cited else float("nan")
    return out


def evaluate_predictions(preds: list[dict], qmap: dict[str, Question], verified: bool,
                         samples: int = 1000, ci: float = 0.95) -> dict:
    per = {"acc_tf": [], "acc_mc": [], "f1_free": [], "f1_long": [], "cit_recall": [], "cit_precision": [],
           "answer_has_valid_citation": []}
    n_cit = n_tree = n_kept = 0
    for p in preds:
        q = qmap[p["question_id"]]
        if q.qtype == "Đúng/Sai":
            per["acc_tf"].append(float(p["pred"] == q.answer))
        elif q.qtype == "Trắc nghiệm":
            per["acc_mc"].append(float(p["pred"] == q.answer))
        elif q.qtype == "Tự luận":
            per["f1_free"].append(token_f1(p["pred"], q.answer))
        else:                                                   # long-form (ViLegalExperts)
            per["f1_long"].append(token_f1(p["answer"], q.answer))
        cs = citation_scores(p, q, verified)
        n_cit, n_tree, n_kept = n_cit + cs["n_citations"], n_tree + cs["in_tree"], n_kept + cs["n_kept"]
        for k in ("cit_recall", "cit_precision"):
            if k in cs and not np.isnan(cs[k]):
                per[k].append(cs[k])
        per["answer_has_valid_citation"].append(float(any(c["status"] == "valid" for c in p["citations"])))
    acc_all = per["acc_tf"] + per["acc_mc"]
    per["acc"] = acc_all
    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in per.items()}
    counts = {f"n_{k}": len(v) for k, v in per.items()}
    cis = {k: bootstrap_ci(v, samples, ci) for k, v in per.items() if v}
    return {"mean": means, "n": counts, "ci95": cis,
            "citations": {"total": n_cit, "kept": n_kept,
                          "share_in_tree": n_tree / n_cit if n_cit else float("nan"),
                          "removed": n_cit - n_kept if verified else 0},
            "mean_prompt_tokens": float(np.mean([p["prompt_tokens"] for p in preds])) if preds else float("nan")}
