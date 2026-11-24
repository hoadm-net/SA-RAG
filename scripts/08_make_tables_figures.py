"""Build Table 1 (main results), Table 2 (ablations) and Figure 1 (quality vs budget) from outputs/metrics.

Run 06_evaluate.py first (it writes outputs/metrics/qa_summary.csv).
Roadmap: Tables and figures
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from sarag.utils.config import resolve_path

SYSTEMS = {"b1_fixed": "B1 fixed", "b2_article": "B2 article", "sarag": "SA-RAG"}
ABLATIONS = {"sarag": "SA-RAG (full)", "ablation_no_breadcrumbs": "w/o breadcrumbs (C1)",
             "ablation_no_aggregation": "w/o aggregation (C2)", "ablation_no_expansion": "w/o expansion (C3)",
             "ablation_no_packing": "w/o budget-aware packing (C4)",
             "ablation_no_citation_verify": "w/o citation verification (C4)"}
ALQAC_EXCLUDED = ["sailor2-1b"]
QA_COLS = {"alqac2024": ["acc", "f1_free", "cit_precision", "cit_recall"],
           "alqac2025": ["acc", "f1_free", "cit_precision", "cit_recall"],
           "vilegalexperts": ["f1_long", "cit_precision", "cit_recall"]}


def retrieval_table(out_dir: Path, runs: dict) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted((out_dir / "metrics" / "retrieval").glob("*")):
        for run, label in runs.items():
            f = ds_dir / f"{run}.json"
            if f.exists():
                m = json.loads(f.read_text())["article"]
                rows.append({"dataset": ds_dir.name, "system": label, "R@1": m["R@1"], "R@5": m["R@5"],
                             "MRR@10": m["MRR@10"]})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f") if not df.empty else "(no results yet)\n"


def paired_tables(out: Path) -> list[str]:
    """Table 3: SA-RAG minus a baseline, per budget, pooled over generators; question-cluster bootstrap CIs."""
    from sarag.evaluation.retrieval import cluster_bootstrap_diff
    from sarag.utils.io import read_jsonl

    def perq(ds, run, gen, b):
        f = out / "metrics" / "qa" / ds / f"{run}__{gen}__B{b}.perq.jsonl"
        return {r["question_id"]: r for r in read_jsonl(f)} if f.exists() else None

    metrics = {"alqac": [("correct", "accuracy")], "vilegalexperts": [("f1", "token F1"), ("judge", "judge score")]}
    rows = []
    for base in ("b2_article", "b1_fixed"):
        for ds_group, ds_list in (("alqac", ["alqac2024", "alqac2025"]), ("vilegalexperts", ["vilegalexperts"])):
            for key, label in metrics[ds_group]:
                for b in (512, 1024, 2048, 4096):
                    diffs: dict = {}
                    gens_used = set()
                    for ds in ds_list:
                        for gen in sorted({f.name.split("__")[1] for f in (out / "metrics" / "qa" / ds).glob(f"sarag__*__B{b}.perq.jsonl")}):
                            if ds.startswith("alqac") and gen in ALQAC_EXCLUDED:
                                continue
                            a, c = perq(ds, "sarag", gen, b), perq(ds, base, gen, b)
                            if not a or not c:
                                continue
                            gens_used.add(gen)
                            for qid, r in a.items():
                                if key in r and qid in c and key in c[qid]:
                                    diffs.setdefault(f"{ds}|{qid}", []).append(r[key] - c[qid][key])
                    if diffs:
                        m, lo, hi = cluster_bootstrap_diff(diffs)
                        rows.append({"vs": SYSTEMS[base], "data": ds_group, "metric": label, "B": b,
                                     "generators": len(gens_used), "questions": len(diffs),
                                     "diff": m, "ci_low": lo, "ci_high": hi,
                                     "significant": "yes" if lo > 0 or hi < 0 else ""})
    return ["\n## Table 3. Paired difference, SA-RAG minus baseline, pooled over generators "
            "(95% CI, bootstrap over questions)\n", to_markdown(pd.DataFrame(rows))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--ablation-generator", default="qwen2.5-3b")
    args = p.parse_args()
    out = resolve_path("outputs")
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    qa = pd.read_csv(out / "metrics" / "qa_summary.csv")
    qa = qa[~qa["run"].str.endswith("_dev")]
    # Sailor2-1B does not follow the answer format on ALQAC (always "Đúng" on true/false, copies the
    # placeholder on multiple choice), so it is left out of ALQAC tables; it stays in ViLegalExperts.
    qa = qa[~(qa["dataset"].str.startswith("alqac") & qa["generator"].isin(ALQAC_EXCLUDED))]

    # Table 1: retrieval + QA + citation for B1, B2, SA-RAG, all generators, at the main budget.
    parts = ["## Table 1a. Article-level retrieval\n", to_markdown(retrieval_table(out, SYSTEMS))]
    main = qa[(qa["budget"] == args.budget) & qa["run"].isin(SYSTEMS)]
    for ds, cols in QA_COLS.items():
        d = main[main["dataset"] == ds]
        if d.empty:
            continue
        d = d.assign(system=d["run"].map(SYSTEMS))[["generator", "system", *cols]].sort_values(["generator", "system"])
        parts += [f"\n## Table 1b. QA and citations, {ds}, B = {args.budget}\n", to_markdown(d)]

    # Table 2: ablations (retrieval for C1/C2, QA for all).
    parts += ["\n## Table 2a. Ablations, retrieval\n",
              to_markdown(retrieval_table(out, {k: v for k, v in ABLATIONS.items() if k in
                                                ("sarag", "ablation_no_breadcrumbs", "ablation_no_aggregation")}))]
    abl = qa[(qa["budget"] == args.budget) & (qa["generator"] == args.ablation_generator) & qa["run"].isin(ABLATIONS)]
    for ds, cols in QA_COLS.items():
        d = abl[abl["dataset"] == ds]
        if not d.empty:
            d = d.assign(variant=d["run"].map(ABLATIONS))[["variant", *cols, "removed", "share_in_tree"]]
            parts += [f"\n## Table 2b. Ablations, {ds} ({args.ablation_generator}, B = {args.budget})\n", to_markdown(d)]
    parts += paired_tables(out)
    (tdir / "tables.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"tables -> {tdir / 'tables.md'}")

    # Figure 1: quality vs budget per generator and system, annotated with local latency / memory if available.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench_f = out / "metrics" / "local_benchmark.jsonl"
    bench = pd.read_json(bench_f, lines=True) if bench_f.exists() else pd.DataFrame()
    sweep = qa[qa["run"].isin(SYSTEMS)]
    datasets = [d for d in QA_COLS if d in set(sweep["dataset"])]
    if datasets:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False)
        styles = {"sarag": "-", "b2_article": "--", "b1_fixed": ":"}
        gens = sorted(sweep["generator"].unique())
        colors = {g: plt.cm.tab10(i) for i, g in enumerate(gens)}       # colour = generator, line style = system
        for ax, ds in zip(axes[0], datasets):
            metric = "acc" if ds.startswith("alqac") else "f1_long"
            d = sweep[sweep["dataset"] == ds]
            for (gen, run), g in d.groupby(["generator", "run"]):
                g = g.sort_values("budget")
                ax.plot(g["budget"], g[metric], styles.get(run, "-"), marker="o", color=colors[gen])
            if not bench.empty:
                for _, r in bench[(bench["setting"] == "gpu") & (bench["generator"] == "qwen2.5-3b")].iterrows():
                    ax.annotate(f"{r['latency_s']:.1f}s, {r['peak_gpu_mib'] / 1024:.1f}GB", (r["budget"], ax.get_ylim()[0]),
                                fontsize=7, ha="center", va="bottom")
            ax.set_xscale("log", base=2)
            ticks = sorted(d["budget"].unique())
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_title(ds)
            ax.set_xlabel("token budget B")
            ax.set_ylabel("accuracy" if metric == "acc" else "token F1")
            ax.grid(alpha=0.3)
        # Shared legend below the panels so it never covers data: colours for generators, line styles for systems.
        from matplotlib.lines import Line2D
        legend = [Line2D([], [], color=colors[g], marker="o", label=g) for g in gens]
        legend += [Line2D([], [], color="black", linestyle=styles[r], label=SYSTEMS[r]) for r in ("sarag", "b2_article", "b1_fixed")]
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fdir = out / "figures"
        fdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fdir / "figure1_budget.png", dpi=200)
        print(f"figure -> {fdir / 'figure1_budget.png'}")


if __name__ == "__main__":
    main()
