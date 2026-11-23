"""Parse each corpus into document trees (C1) and report parse success / fallback rates.

Roadmap: Parser
"""

import json
import re
from pathlib import Path

from sarag.parsing.loaders import iter_documents, save_documents
from sarag.utils.cli import base_parser, setup
from sarag.utils.config import resolve_path

ALL = ["alqac2024", "alqac2025", "zalo", "vilegalexperts"]
REF_RE = re.compile(r"(?:khoản\s+(\d+)\s+)?Điều\s+(\d+[a-zđ]?)", re.I)


def report(docs) -> dict:
    arts = [a for d in docs for a in d.articles]
    real = [a for d in docs if not d.fallback for a in d.articles]
    return {
        "documents": len(docs),
        "fallback_documents": sum(d.fallback for d in docs),
        "fallback_rate": round(sum(d.fallback for d in docs) / max(len(docs), 1), 4),
        "articles": len(real),
        "articles_with_title": sum(bool(a.title) for a in real),
        "articles_with_lead_in": sum(bool(a.lead_in) for a in real),
        "articles_with_clauses": sum(bool(a.clauses) for a in real),
        "clauses": sum(len(a.clauses) for a in arts),
        "points": sum(len(a.points) + sum(len(c.points) for c in a.clauses) for a in arts),
        "articles_with_chapter": sum(bool(a.chapter) for a in real),
        "rejected_numbered_lines": sum(a.rejected_lines for a in arts),
        "clean_article_rate": round(sum(a.rejected_lines == 0 for a in real) / max(len(real), 1), 4),
    }


def gold_coverage(name: str, ds_cfg: dict, raw: Path, docs) -> dict:
    """Share of gold references that resolve to a parsed article (and clause, when given)."""
    by_id = {d.doc_id: d for d in docs}
    hit = total = clause_hit = clause_total = 0
    if ds_cfg["format"] == "article_level":
        for p in ds_cfg["test"]:
            for q in json.loads((raw / p).read_text(encoding="utf-8")):
                for r in q["relevant_articles"]:
                    total += 1
                    d = by_id.get(r["law_id"])
                    hit += bool(d and d.article(str(r["article_id"])))
    else:  # silver references parsed from expert answers, checked against the question's own document
        for q in json.loads((raw / ds_cfg["test"][0]).read_text(encoding="utf-8")).values():
            ctx = next(iter(q["contexts"].values()))
            d = by_id.get(Path(ctx["content"]).stem)
            if d is None:
                continue
            for k, n in set(REF_RE.findall(q["answer"])):
                total += 1
                art = d.article(n)
                hit += art is not None
                if k:
                    clause_total += 1
                    clause_hit += bool(art and art.clause(int(k)))
    out = {"gold_refs": total, "resolved_article_rate": round(hit / max(total, 1), 4)}
    if clause_total:
        out.update(clause_refs=clause_total, resolved_clause_rate=round(clause_hit / clause_total, 4))
    return out


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    args = p.parse_args()
    cfg = setup(args)
    raw = resolve_path(cfg["paths"]["raw"])
    processed = resolve_path(cfg["paths"]["processed"])
    for name in args.datasets:
        ds_cfg = cfg["datasets"][name]
        docs = list(iter_documents(name, ds_cfg, raw))
        save_documents(docs, processed / name / "trees.jsonl")
        rep = {**report(docs), **gold_coverage(name, ds_cfg, raw, docs)}
        (processed / name / "parse_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in rep.items()))


if __name__ == "__main__":
    main()
