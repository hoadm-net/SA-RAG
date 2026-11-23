"""Build index units (clause / article / fixed, with or without breadcrumbs) and their BM25 and bge-m3 indexes.

Roadmap: Indexes
"""

import json
import time

from sarag.indexing.bm25 import BM25Index
from sarag.indexing.units import build_units
from sarag.parsing.loaders import load_documents
from sarag.utils.cli import base_parser, setup
from sarag.utils.config import load_models, resolve_path
from sarag.utils.io import write_jsonl

ALL = ["alqac2024", "alqac2025", "zalo", "vilegalexperts"]
# variant name -> (unit type, breadcrumbs). Used by: SA-RAG, no-breadcrumb ablation, B2, B1.
VARIANTS = {"clause_bc": ("clause", True), "clause": ("clause", False), "article": ("article", False),
            "fixed": ("fixed", False)}


def refresh_units(out, docs, variant, icfg, count, tok) -> None:
    """Rewrite unit metadata (e.g. article mapping) without touching BM25 / dense indexes."""
    unit, crumbs = VARIANTS[variant]
    max_tok = icfg["max_article_tokens"] if unit == "article" else icfg["max_unit_tokens"]
    units = build_units(docs, unit, crumbs, count, max_tok, icfg["fixed_chunk"]["size"],
                        icfg["fixed_chunk"]["overlap"], tok.tokenize)
    with open(out / "units.jsonl", encoding="utf-8") as f:
        old = [json.loads(line) for line in f]
    new_texts = [u.text for u in units]
    old_texts = [f"{u['breadcrumb']}\n{u['body']}" if u["breadcrumb"] else u["body"] for u in old]
    if new_texts != old_texts:
        raise SystemExit(f"{out}: unit texts changed; rebuild the index with --force instead")
    write_jsonl(out / "units.jsonl", (u.to_dict() for u in units))
    print(f"[{out.parent.parent.name}/{variant}] units.jsonl refreshed ({len(units)} units)")


def main() -> None:
    p = base_parser(__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=ALL, choices=ALL)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--skip-dense", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild even if the index exists")
    p.add_argument("--units-only", action="store_true",
                   help="rewrite units.jsonl only; unit texts must match the existing index (no re-encoding)")
    args = p.parse_args()
    cfg = setup(args)
    dense_cfg = load_models()["retrievers"]["dense"]
    icfg = cfg["indexing"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dense_cfg["hf_id"])
    count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    encoder = None

    for name in args.datasets:
        docs = load_documents(resolve_path(cfg["paths"]["processed"]) / name / "trees.jsonl")
        for variant in args.variants:
            out = resolve_path(cfg["paths"]["processed"]) / name / "index" / variant
            if args.units_only:
                refresh_units(out, docs, variant, icfg, count, tok)
                continue
            if (out / "meta.json").exists() and not args.force:
                print(f"[{name}/{variant}] exists, skipping")
                continue
            out.mkdir(parents=True, exist_ok=True)
            unit, crumbs = VARIANTS[variant]
            t0 = time.time()
            # B2 keeps whole articles (split only beyond the largest budget); other units fit the encoder window.
            max_tok = icfg["max_article_tokens"] if unit == "article" else icfg["max_unit_tokens"]
            units = build_units(docs, unit, crumbs, count, max_tok,
                                icfg["fixed_chunk"]["size"], icfg["fixed_chunk"]["overlap"], tok.tokenize)
            write_jsonl(out / "units.jsonl", (u.to_dict() for u in units))
            texts = [u.text for u in units]
            BM25Index.build(texts, icfg["segmenter"]).save(out / "bm25")
            t1 = time.time()
            if not args.skip_dense:
                if encoder is None:
                    from sarag.indexing.dense import DenseEncoder
                    encoder = DenseEncoder(dense_cfg["hf_id"], dense_cfg["max_length"], dense_cfg["batch_size"],
                                           dense_cfg["fp16"])
                from sarag.indexing.dense import DenseIndex
                DenseIndex(encoder.encode(texts, max_length=max(max_tok, dense_cfg["max_length"]))).save(out / "dense.npy")
            ntok = sorted(u.n_tokens for u in units)
            meta = {"dataset": name, "variant": variant, "unit": unit, "breadcrumbs": crumbs, "units": len(units),
                    "split_pieces": sum("#" in u.unit_id for u in units),
                    "tokens": {"median": ntok[len(ntok) // 2], "p95": ntok[int(len(ntok) * .95)], "max": ntok[-1]},
                    "dense": not args.skip_dense, "dense_model": dense_cfg["hf_id"],
                    "seconds": {"units_bm25": round(t1 - t0, 1), "dense": round(time.time() - t1, 1)}}
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[{name}/{variant}] " + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
