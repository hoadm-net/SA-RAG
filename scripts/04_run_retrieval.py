"""Run hybrid retrieval + RRF + article aggregation for each system; write rankings and article-level metrics.

Roadmap: Retrieval
"""

import json
from sarag.datasets import load_questions
from sarag.evaluation.retrieval import bootstrap_ci, retrieval_metrics
from sarag.parsing.loaders import load_documents
from sarag.retrieval.hybrid import HybridRetriever
from sarag.utils.seed import set_seed
from sarag.utils.config import load_config, load_models, resolve_path
from sarag.utils.io import write_jsonl

ALL = ["alqac2024", "alqac2025", "zalo", "vilegalexperts"]
SYSTEMS = ["sarag", "baseline_b1_fixed", "baseline_b2_article", "ablation_no_breadcrumbs", "ablation_no_aggregation"]


def index_variant(cfg: dict) -> str:
    unit = cfg["indexing"]["unit"]
    if unit == "clause":
        return "clause_bc" if cfg["indexing"]["breadcrumbs"] else "clause"
    return unit


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--systems", nargs="+", default=SYSTEMS, help="experiment config names")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for every system")
    p.add_argument("--tag", default="", help="suffix for output names (use with --set)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--split", default="test", choices=["test", "dev"], help="dev is for tuning only")
    args = p.parse_args()

    dense_cfg = load_models()["retrievers"]["dense"]
    encoder = None
    for name in args.datasets:
        for system in args.systems:
            cfg = load_config(f"configs/experiments/{system}.yaml", args.set)
            set_seed(cfg["seed"])
            rcfg, ecfg = cfg["retrieval"], cfg["evaluation"]
            ds_cfg = cfg["datasets"][name]
            processed = resolve_path(cfg["paths"]["processed"]) / name
            docs = load_documents(processed / "trees.jsonl")
            questions = load_questions(name, ds_cfg, resolve_path(cfg["paths"]["raw"]), docs, args.split)

            if rcfg["dense"] and encoder is None:
                from sarag.indexing.dense import DenseEncoder
                encoder = DenseEncoder(dense_cfg["hf_id"], dense_cfg["max_length"], dense_cfg["batch_size"],
                                       dense_cfg["fp16"])
            variant = index_variant(cfg)
            retriever = HybridRetriever(processed / "index" / variant, rcfg["bm25"], rcfg["dense"],
                                        encoder if rcfg["dense"] else None)
            results = []
            for i in range(0, len(questions), args.batch):
                results += retriever.search([q.text for q in questions[i:i + args.batch]], rcfg["top_k_units"],
                                            rcfg["rrf_k"], rcfg["aggregation"], rcfg["candidates"],
                                            rcfg["bm25_weight"], rcfg["dense_weight"])

            run = f"{cfg['name']}{args.tag}" + ("_dev" if args.split == "dev" else "")
            out = resolve_path(cfg["paths"]["outputs"]) / "retrieval" / name
            write_jsonl(out / f"{run}.jsonl", ({
                "question_id": q.qid, "config": run, "index": variant,
                "gold_articles": [list(g) for g in q.gold_articles], "gold_docs": q.gold_docs,
                "articles": [[*a, round(s, 6)] for a, s in r.articles[:rcfg["top_k_articles"] * 5]],
                "units": [[retriever.units[u]["unit_id"], round(s, 6)] for u, s in r.units],
            } for q, r in zip(questions, results)))

            ranked = [[a for a, _ in r.articles] for r in results]
            golds = [set(q.gold_articles) for q in questions]
            m = retrieval_metrics(ranked, golds, ecfg["retrieval_k"], ecfg["mrr_k"])
            metrics = {"dataset": name, "system": run, "index": variant, "silver": any(q.silver for q in questions),
                       "article": m["mean"],
                       "article_ci95": {k: bootstrap_ci(v, ecfg["bootstrap_samples"], ecfg["ci"])
                                        for k, v in m["per_question"].items()}}
            if name == "vilegalexperts":            # document-level retrieval against the gold document
                doc_ranked = [list(dict.fromkeys(a[0] for a in rk)) for rk in ranked]
                metrics["document"] = retrieval_metrics(doc_ranked, [set(q.gold_docs) for q in questions],
                                                        ecfg["retrieval_k"], ecfg["mrr_k"])["mean"]
            mdir = resolve_path(cfg["paths"]["outputs"]) / "metrics" / "retrieval" / name
            mdir.mkdir(parents=True, exist_ok=True)
            (mdir / f"{run}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            a = metrics["article"]
            line = f"[{name}] {run:28s} n={a['n']:<5} " + " ".join(f"{k}={v:.3f}" for k, v in a.items() if k != "n")
            if "document" in metrics:
                line += " | doc " + " ".join(f"{k}={v:.3f}" for k, v in metrics["document"].items() if k != "n")
            print(line, flush=True)


if __name__ == "__main__":
    main()
