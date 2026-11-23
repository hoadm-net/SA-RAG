"""Download ALQAC 2024/2025, Zalo Legal 2021 and ViLegalExperts into data/raw; print dataset statistics.

Roadmap: Data
"""

import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError

from sarag.utils.cli import base_parser, setup
from sarag.utils.config import resolve_path
from sarag.utils.env import require_env

ALQAC_REPO = "nguyenlab/ALQAC"
VLE_REPO = "DatNguyenAIML/ViLegalExperts"
ALL = ["alqac2024", "alqac2025", "zalo", "vilegalexperts"]

CLAUSE_RE = re.compile(r"^\s*\d+\.\s", re.M)
POINT_RE = re.compile(r"^\s*[a-zđ]\)\s", re.M)
CHAPTER_RE = re.compile(r"^\s*Chương\s+[IVXLC]+", re.M)
ARTICLE_RE = re.compile(r"^\s*Điều\s+\d+[a-z]?\.", re.M)
XREF_RE = re.compile(r"khoản\s+\d+.{0,20}Điều\s+\d+|Điều\s+\d+\s+của\s+(Luật|Bộ luật|Nghị định)\s+này", re.I)
ANSWER_REF_RE = re.compile(r"Điều\s+\d+", re.I)


# ----------------------------------------------------------------------------- download

def _snapshot(repo: str, out: Path, patterns: list[str]) -> None:
    try:
        snapshot_download(repo, repo_type="dataset", token=require_env("HF_TOKEN"),
                          local_dir=out, allow_patterns=patterns)
    except GatedRepoError:
        sys.exit(f"No access to {repo}: accept its terms at https://huggingface.co/datasets/{repo} "
                 "with the account that owns HF_TOKEN, then rerun.")


def download(name: str, raw: Path) -> None:
    if name in ("alqac2024", "alqac2025", "zalo"):
        sub = {"alqac2024": "ALQAC_2024/*", "alqac2025": "ALQAC_2025/*", "zalo": "additional_data/zalo/*"}[name]
        _snapshot(ALQAC_REPO, raw / "alqac", ["README.md", sub])
    elif name == "vilegalexperts":
        # Test split + every released context document (~0.26 GB); train/dev are not needed (training-free).
        _snapshot(VLE_REPO, raw / "vilegalexperts", ["README.md", "QA_single_context/test_data.json", "contexts/*"])
    else:
        raise ValueError(name)
    print(f"downloaded {name}")


# ----------------------------------------------------------------------------- statistics

def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def corpus_stats(law: list[dict]) -> dict:
    """Stats for an article-level corpus: [{id, articles: [{id, text}]}]."""
    arts = [a for doc in law for a in doc["articles"]]
    words = [len(a["text"].split()) for a in arts]
    n_cl = [len(CLAUSE_RE.findall(a["text"])) for a in arts]
    n_pt = [len(POINT_RE.findall(a["text"])) for a in arts]
    missing = {}
    for doc in law:
        ids = [int(a["id"]) for a in doc["articles"] if a["id"].isdigit()]
        gap = sorted(set(range(1, max(ids) + 1)) - set(ids)) if ids else []
        if gap:
            missing[doc["id"]] = gap
    return {
        "documents": len(law), "articles": len(arts),
        "words_per_article": {"median": st.median(words), "mean": round(st.mean(words), 1), "max": max(words)},
        "articles_with_clauses": sum(c > 0 for c in n_cl), "clauses": sum(n_cl),
        "articles_with_points": sum(p > 0 for p in n_pt), "points": sum(n_pt),
        "articles_with_cross_refs": sum(bool(XREF_RE.search(a["text"])) for a in arts),
        "missing_article_ids": missing,
    }


def question_stats(qs: list[dict], keys: set) -> dict:
    rel = [len(q["relevant_articles"]) for q in qs]
    return {
        "n": len(qs), "types": dict(collections.Counter(q.get("question_type", "-") for q in qs)),
        "gold_per_question": {"mean": round(st.mean(rel), 3), "max": max(rel)},
        "gold_missing_from_corpus": sum((r["law_id"], r["article_id"]) not in keys
                                        for q in qs for r in q["relevant_articles"]),
    }


def stats_article_level(law_path: Path, splits: dict[str, Path]) -> dict:
    law = _load(law_path)
    keys = {(doc["id"], a["id"]) for doc in law for a in doc["articles"]}
    return {"corpus": corpus_stats(law), "questions": {s: question_stats(_load(p), keys) for s, p in splits.items()}}


def stats_vilegalexperts(d: Path) -> dict:
    qs = list(_load(d / "QA_single_context" / "test_data.json").values())
    ctx_dir = d / "contexts"
    released = {p.name for p in ctx_dir.glob("context_*.json")}
    covered = [q for q in qs if all(c["content"] in released for c in q["contexts"].values())]
    n_ch = n_ar = n_cl = 0
    for p in ctx_dir.glob("context_*.json"):
        text = _load(p)["passage"]
        n_ch += len(CHAPTER_RE.findall(text))
        n_ar += len(ARTICLE_RE.findall(text))
        n_cl += len(CLAUSE_RE.findall(text))
    return {
        "corpus": {"documents": len(released), "chapters": n_ch, "articles": n_ar, "clauses": n_cl},
        "questions": {"test_all": len(qs), "test_with_context": len(covered),
                      "with_article_ref_in_answer": sum(bool(ANSWER_REF_RE.search(q["answer"])) for q in covered),
                      "answer_words_median": st.median(len(q["answer"].split()) for q in covered),
                      "gold_doc_types": dict(collections.Counter(q["most_common_doc_type"] for q in covered).most_common())},
    }


def compute_stats(name: str, raw: Path) -> dict:
    a = raw / "alqac"
    if name == "alqac2024":
        d = a / "ALQAC_2024"
        return stats_article_level(d / "law.json", {s: d / f"{s}.json" for s in
                                   ["train", "unverified_train", "public_test_GOLD", "private_test_GOLD"]})
    if name == "alqac2025":
        d = a / "ALQAC_2025"
        return stats_article_level(d / "law.json", {s: d / f"{s}.json" for s in ["train", "private_test_GOLD"]})
    if name == "zalo":
        d = a / "additional_data" / "zalo"
        return stats_article_level(d / "zalo_corpus.json", {"all": d / "zalo_question.json"})
    if name == "vilegalexperts":
        return stats_vilegalexperts(raw / "vilegalexperts")
    raise ValueError(name)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()
    cfg = setup(args)
    raw = resolve_path(cfg["paths"]["raw"])
    out_dir = resolve_path(cfg["paths"]["outputs"]) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if not args.skip_download:
            download(name, raw)
        stats = compute_stats(name, raw)
        (out_dir / f"{name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        c = stats["corpus"]
        print(f"[{name}] corpus: " + ", ".join(f"{k}={v}" for k, v in c.items() if isinstance(v, int)))
        for k, v in stats["questions"].items():
            print(f"[{name}]   {k}: {v if not isinstance(v, dict) else {x: v[x] for x in list(v)[:3]}}")
    print(f"stats -> {out_dir}")


if __name__ == "__main__":
    main()
