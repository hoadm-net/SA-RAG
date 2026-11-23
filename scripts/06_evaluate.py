"""Compute QA and citation metrics (with bootstrap CIs) for every prediction file; optional LLM judge.

Roadmap: Evaluation
"""

import argparse
import csv
import json
import re
from pathlib import Path

from sarag.datasets import load_questions
from sarag.evaluation.qa import evaluate_predictions
from sarag.parsing.loaders import load_documents
from sarag.utils.config import load_config, load_models, resolve_path
from sarag.utils.io import read_jsonl

SUMMARY_KEYS = ["acc", "acc_tf", "acc_mc", "f1_free", "f1_long", "cit_precision", "cit_recall",
                "answer_has_valid_citation"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=["alqac2024", "alqac2025", "vilegalexperts"])
    p.add_argument("--pattern", default="*.jsonl", help="glob over outputs/predictions/<dataset>/")
    p.add_argument("--judge", action="store_true", help="also run the LLM judge (costs money)")
    p.add_argument("--judge-limit", type=int, default=0, help="judge only the first N eligible questions (pilot)")
    args = p.parse_args()

    base = load_config("configs/default.yaml")
    rows = []
    for name in args.datasets:
        files = sorted((resolve_path(base["paths"]["outputs"]) / "predictions" / name).glob(args.pattern))
        if not files:
            continue
        docs = load_documents(resolve_path(base["paths"]["processed"]) / name / "trees.jsonl")
        qmaps = {}
        for f in files:
            run, gen, budget = f.stem.split("__")
            split = "dev" if run.endswith("_dev") else "test"
            if split not in qmaps:
                qmaps[split] = {q.qid: q for q in load_questions(name, base["datasets"][name],
                                                                 resolve_path(base["paths"]["raw"]), docs, split)}
            system = re.sub(r"_v\d+$", "", run.removesuffix("_dev"))   # drop prompt-version tags (_v0, _v1)
            cfg_file = resolve_path(f"configs/experiments/{system}.yaml")
            verified = (load_config(cfg_file)["generation"]["verify_citations"] if cfg_file.exists()
                        else base["generation"]["verify_citations"])
            preds = list(read_jsonl(f))
            m = evaluate_predictions(preds, qmaps[split], verified, base["evaluation"]["bootstrap_samples"],
                                     base["evaluation"]["ci"])
            if args.judge:
                m["judge"] = run_judge(preds, qmaps[split], base, args.judge_limit)
            out = resolve_path(base["paths"]["outputs"]) / "metrics" / "qa" / name / f"{f.stem}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
            row = {"dataset": name, "run": run, "generator": gen, "budget": int(budget[1:]), "n": len(preds),
                   **{k: round(m["mean"][k], 4) for k in SUMMARY_KEYS},
                   "share_in_tree": round(m["citations"]["share_in_tree"], 4),
                   "citations": m["citations"]["total"], "removed": m["citations"]["removed"],
                   "prompt_tokens": round(m["mean_prompt_tokens"], 1)}
            if "judge" in m:
                row.update(judge_score=m["judge"]["mean_score"], judge_correct=m["judge"]["correct_rate"])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    if rows:
        out = resolve_path(base["paths"]["outputs"]) / "metrics" / "qa_summary.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"summary -> {out}")


def run_judge(preds, qmap, base, limit: int = 0) -> dict:
    """Judge free-text answers: long-form answers (ViLegalExperts) and 'Tự luận' questions (ALQAC)."""
    from concurrent.futures import ThreadPoolExecutor

    from sarag.evaluation.judge import Judge

    jcfg = load_models()["judge"]
    judge = Judge(jcfg["model"], jcfg["base_url"], jcfg["api_key_env"], resolve_path(base["paths"]["cache"]),
                  jcfg["temperature"])
    todo = [p for p in preds if qmap[p["question_id"]].qtype in ("", "Tự luận")]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"n": 0, "mean_score": float("nan"), "correct_rate": float("nan")}

    def one(p):
        q = qmap[p["question_id"]]
        return judge.score(q.text, q.answer, p["answer"])

    with ThreadPoolExecutor(jcfg.get("workers", 16)) as ex:
        scores = list(ex.map(one, todo))
    valid = [s for s in scores if 1 <= s["score"] <= 5]
    return {"n": len(todo), "n_unparsed": len(todo) - len(valid), "model": jcfg["model"],
            "mean_score": sum(s["score"] for s in valid) / len(valid) if valid else float("nan"),
            "correct_rate": sum(s["correct"] for s in valid) / len(valid) if valid else float("nan")}


if __name__ == "__main__":
    main()
