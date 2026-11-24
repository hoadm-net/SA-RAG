"""Expand (C3), pack to the token budget (C4) and generate cited answers (resumable JSONL).

Stages:
  pack      build the context for every (dataset, system, generator, budget) and report context recall
  generate  run the generator on packed contexts (vLLM), parse and verify citations
Roadmap: Generation
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from sarag.datasets import load_questions
from sarag.expansion.structural import Candidate, expand
from sarag.generation.prompts import PromptBuilder
from sarag.indexing.store import UnitStore
from sarag.parsing.loaders import load_documents
from sarag.packing.packer import Packer
from sarag.utils.config import load_config, load_models, resolve_path
from sarag.generation.citations import parse_answer, parse_citations, strip_invalid, verify
from sarag.utils.io import done_ids, read_jsonl, write_jsonl
from sarag.utils.seed import set_seed

ALL = ["alqac2024", "alqac2025", "zalo", "vilegalexperts"]
SYSTEMS = ["sarag", "baseline_b1_fixed", "baseline_b2_article"]
SAFETY_TOKENS = 8          # chat-template joins can shift a token or two at the context boundary


def index_variant(cfg: dict) -> str:
    unit = cfg["indexing"]["unit"]
    if unit == "clause":
        return "clause_bc" if cfg["indexing"]["breadcrumbs"] else "clause"
    return unit


def candidates(rec: dict, store: UnitStore, cfg: dict) -> list[Candidate]:
    """Retrieved units as packing candidates. With article aggregation, a unit's priority blends its own
    score with its article's aggregated (max) score, so clauses of strong articles are packed together."""
    lam = cfg["packing"]["article_weight"] if cfg["retrieval"]["aggregation"] != "none" else 0.0
    units = [(store.by_id[uid], s) for uid, s in rec["units"] if uid in store.by_id]
    art_best: dict = defaultdict(float)
    for i, s in units:
        for a in store.units[i]["articles"]:
            key = (store.units[i]["doc_id"], store.units[i]["part"], a)
            art_best[key] = max(art_best[key], s)
    out = []
    for i, s in units:
        u = store.units[i]
        a = max((art_best[(u["doc_id"], u["part"], x)] for x in u["articles"]), default=s)  # empty-text windows map to no article
        out.append(Candidate(i, lam * a + (1 - lam) * s))
    return out


def pack_stage(name, system, gen_key, budget, cfg, models, tag, split):
    pcfg, xcfg = cfg["packing"], cfg["expansion"]
    run = f"{cfg['name']}{tag}" + ("_dev" if split == "dev" else "")
    retr_name = cfg.get("retrieval_from", cfg["name"])        # ablations of C3/C4 reuse SA-RAG retrieval
    retr_path = resolve_path(cfg["paths"]["outputs"]) / "retrieval" / name / f"{retr_name}{'_dev' if split == 'dev' else ''}.jsonl"
    processed = resolve_path(cfg["paths"]["processed"]) / name
    store = UnitStore.load(processed, index_variant(cfg))
    docs = list(store.docs.values())
    qmap = {q.qid: q for q in load_questions(name, cfg["datasets"][name], resolve_path(cfg["paths"]["raw"]), docs, split)}

    from transformers import AutoTokenizer
    gcfg = models["generators"][gen_key]
    pb = PromptBuilder(AutoTokenizer.from_pretrained(gcfg["hf_id"]),
                       False if gcfg.get("enable_thinking") is False else None,
                       cfg["generation"]["prompt"])
    packer = Packer(store, pb.count, lead_in=xcfg["enabled"] and xcfg["parent_lead_in"], truncate=pb.truncate)

    out_rows, recalls, used, over = [], [], [], 0
    for rec in read_jsonl(retr_path):
        q = qmap[rec["question_id"]]
        cands = candidates(rec, store, cfg)
        if xcfg["enabled"] and xcfg["cross_refs"]:
            cands = expand(cands, store, xcfg["xref_decay"], xcfg["max_targets"])
        room = pb.context_budget(q, budget) - SAFETY_TOKENS
        packed = packer.pack(cands, room, pcfg["strategy"], pcfg["length_exponent"], pcfg["max_candidates"])
        prompt_tokens = pb.count(pb.render(q, packed.text))
        over += prompt_tokens > budget
        gold = {tuple(g) for g in rec["gold_articles"]}
        if gold:
            recalls.append(len(gold & set(packed.articles)) / len(gold))
        used.append(prompt_tokens)
        out_rows.append({"question_id": q.qid, "config": run, "generator": gen_key, "budget": budget,
                         "retrieved_ids": [a[:3] for a in rec["articles"][:cfg["retrieval"]["top_k_articles"]]],
                         "context_ids": [store.units[i]["unit_id"] for i in packed.units],
                         "context_articles": [list(a) for a in packed.articles], "context": packed.text,
                         "blocks": [{k: v for k, v in b.items() if k != "units"} for b in packed.blocks],
                         "prompt_tokens": prompt_tokens, "truncated_unit": packed.truncated_unit is not None})
    out = resolve_path(cfg["paths"]["outputs"]) / "contexts" / name / f"{run}__{gen_key}__B{budget}.jsonl"
    write_jsonl(out, out_rows)
    stats = {"dataset": name, "system": run, "generator": gen_key, "budget": budget, "n": len(out_rows),
             "context_article_recall": float(np.mean(recalls)) if recalls else float("nan"),
             "mean_prompt_tokens": float(np.mean(used)), "over_budget": over,
             "mean_context_articles": float(np.mean([len(r["context_articles"]) for r in out_rows]))}
    print(f"[pack] {name} {run:24s} {gen_key:13s} B={budget:<5} recall={stats['context_article_recall']:.3f} "
          f"tokens={stats['mean_prompt_tokens']:.0f} articles={stats['mean_context_articles']:.1f} over={over}", flush=True)
    return stats


def generate_stage(name, system, gen_key, budget, cfg, models, tag, split, engine_cache):
    gcfg_all = cfg["generation"]
    run = f"{cfg['name']}{tag}" + ("_dev" if split == "dev" else "")
    ctx_path = resolve_path(cfg["paths"]["outputs"]) / "contexts" / name / f"{run}__{gen_key}__B{budget}.jsonl"
    if not ctx_path.exists():
        pack_stage(name, system, gen_key, budget, cfg, models, tag, split)
    out = resolve_path(cfg["paths"]["outputs"]) / "predictions" / name / f"{run}__{gen_key}__B{budget}.jsonl"
    done = done_ids(out)
    rows = [r for r in read_jsonl(ctx_path) if r["question_id"] not in done]
    if not rows:
        print(f"[gen] {name} {run} {gen_key} B={budget}: complete", flush=True)
        return
    processed = resolve_path(cfg["paths"]["processed"]) / name
    docs = load_documents(processed / "trees.jsonl")
    doc_map = {d.doc_id: d for d in docs}
    qmap = {q.qid: q for q in load_questions(name, cfg["datasets"][name], resolve_path(cfg["paths"]["raw"]), docs, split)}

    gcfg = models["generators"][gen_key]
    if gen_key not in engine_cache:
        engine_cache.clear()                                  # one engine on the GPU at a time
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        from sarag.generation.backends import VLLMGenerator
        engine_cache[gen_key] = VLLMGenerator(gcfg["hf_id"], max_model_len=gcfg.get("max_model_len", 6144),
                                              gpu_memory_utilization=gcfg.get("gpu_memory_utilization", 0.85),
                                              max_new_tokens=gcfg_all["max_new_tokens"], seed=cfg["seed"],
                                              **gcfg.get("engine_kwargs", {}))
    engine = engine_cache[gen_key]
    from transformers import AutoTokenizer
    pb = PromptBuilder(AutoTokenizer.from_pretrained(gcfg["hf_id"]),
                       False if gcfg.get("enable_thinking") is False else None,
                       cfg["generation"]["prompt"])
    prompts = [pb.render(qmap[r["question_id"]], r["context"]) for r in rows]
    texts, latency = engine.generate(prompts)
    preds = []
    for r, text in zip(rows, texts):
        q = qmap[r["question_id"]]
        cites = verify(parse_citations(text), r["blocks"], doc_map)
        answer = strip_invalid(text, cites) if gcfg_all["verify_citations"] else text
        preds.append({"question_id": q.qid, "config": run, "generator": gen_key, "budget": budget,
                      "retrieved_ids": r["retrieved_ids"], "context_ids": r["context_ids"],
                      "answer": answer, "raw_output": text, "pred": parse_answer(text, q.qtype),
                      "citations": [c.to_dict() for c in cites],
                      "n_invalid_citations": sum(c.status != "valid" for c in cites),
                      "prompt_tokens": r["prompt_tokens"], "latency": latency})
    write_jsonl(out, preds, append=True)
    n_c = sum(len(p["citations"]) for p in preds)
    n_bad = sum(p["n_invalid_citations"] for p in preds)
    print(f"[gen] {name} {run:24s} {gen_key:13s} B={budget:<5} n={len(preds)} citations={n_c} invalid={n_bad} "
          f"s/q={latency:.2f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=["pack", "generate"], default="pack")
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--systems", nargs="+", default=SYSTEMS)
    p.add_argument("--generators", nargs="+", default=["qwen2.5-3b"])
    p.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--tag", default="")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    args = p.parse_args()
    models = load_models()
    all_stats, engines = [], {}
    # Generator outermost so each model is loaded on the GPU once.
    for gen in args.generators:
        for name in args.datasets:
            for system in args.systems:
                cfg = load_config(f"configs/experiments/{system}.yaml", args.set)
                set_seed(cfg["seed"])
                if name == "zalo" and args.stage == "generate":
                    continue                              # retrieval-only dataset
                for b in args.budgets:
                    window = models["generators"][gen].get("max_model_len", 6144)
                    if b + cfg["generation"]["max_new_tokens"] > window:
                        print(f"[skip] {name} {system} {gen} B={b}: budget + answer exceeds the {window}-token window",
                              flush=True)
                        continue
                    if args.stage == "pack":
                        all_stats.append(pack_stage(name, system, gen, b, cfg, models, args.tag, args.split))
                    else:
                        generate_stage(name, system, gen, b, cfg, models, args.tag, args.split, engines)
    if all_stats:
        mdir = resolve_path("outputs/metrics/packing")
        mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / "packing_stats.jsonl", "a", encoding="utf-8") as f:
            for s in all_stats:
                f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
