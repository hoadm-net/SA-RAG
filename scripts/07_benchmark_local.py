"""Benchmark GGUF Q4_K_M generators with llama.cpp: TTFT, tok/s, latency, peak memory.

Uses packed SA-RAG contexts (outputs/contexts/...) so prompts match the main experiments.
Two settings: CPU only (laptop proxy, limited threads) and full GPU offload.
Roadmap: Local benchmarks
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import psutil

from sarag.datasets import load_questions
from sarag.generation.prompts import PromptBuilder
from sarag.parsing.loaders import load_documents
from sarag.utils.config import load_config, load_models, resolve_path
from sarag.utils.io import read_jsonl

GGUF_DIR = Path("/workspace/models/gguf")


def gpu_mem_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


class PeakSampler:
    """Samples process RSS and GPU memory every 50 ms."""

    def __init__(self):
        self.proc = psutil.Process()
        self.rss = self.gpu = 0
        self._stop = threading.Event()

    def __enter__(self):
        self.base_gpu = gpu_mem_mib()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.rss = max(self.rss, self.proc.memory_info().rss)
            self.gpu = max(self.gpu, gpu_mem_mib() - self.base_gpu)
            time.sleep(0.05)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def bench_one(llm, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    ttft, n = None, 0
    for chunk in llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0, stream=True):
        if ttft is None:
            ttft = time.perf_counter() - t0
        n += 1
    total = time.perf_counter() - t0
    decode = total - (ttft or 0)
    return {"ttft_s": ttft or total, "latency_s": total, "new_tokens": n,
            "tok_per_s": (n - 1) / decode if n > 1 and decode > 0 else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--generators", nargs="+", default=["qwen2.5-1.5b", "qwen2.5-3b", "qwen3-1.7b", "sailor2-1b", "qwen2.5-7b"])
    p.add_argument("--budgets", nargs="+", type=int, default=[1024, 2048, 4096])
    p.add_argument("--dataset", default="alqac2024")
    p.add_argument("--n", type=int, default=20, help="questions per setting")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--cpu-threads", type=int, default=8, help="CPU-only setting, laptop proxy")
    p.add_argument("--settings", nargs="+", default=["cpu", "gpu"])
    args = p.parse_args()

    from llama_cpp import Llama
    from transformers import AutoTokenizer

    cfg = load_config("configs/experiments/sarag.yaml")
    models = load_models()
    docs = load_documents(resolve_path(cfg["paths"]["processed"]) / args.dataset / "trees.jsonl")
    qmap = {q.qid: q for q in load_questions(args.dataset, cfg["datasets"][args.dataset], resolve_path(cfg["paths"]["raw"]), docs)}
    out_path = resolve_path(cfg["paths"]["outputs"]) / "metrics" / "local_benchmark.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for gen in args.generators:
        gguf = GGUF_DIR / f"{models['generators'][gen]['hf_id'].split('/')[1].lower()}-q4_k_m.gguf"
        if not gguf.exists():
            print(f"[skip] {gen}: {gguf} not found")
            continue
        gcfg = models["generators"][gen]
        pb = PromptBuilder(AutoTokenizer.from_pretrained(gcfg["hf_id"]),
                           False if gcfg.get("enable_thinking") is False else None, cfg["generation"]["prompt"])
        for setting in args.settings:
            kw = {"n_gpu_layers": -1} if setting == "gpu" else {"n_gpu_layers": 0, "n_threads": args.cpu_threads}
            with PeakSampler() as mem:
                llm = Llama(model_path=str(gguf), n_ctx=min(max(args.budgets) + args.max_tokens + 64,
                                                             gcfg.get("max_model_len", 1 << 20)), verbose=False,
                            seed=cfg["seed"], **kw)
                for b in args.budgets:
                    if b + args.max_tokens > gcfg.get("max_model_len", 6144):
                        continue                                  # e.g. Sailor2 has a 4k window
                    ctx_file = resolve_path(cfg["paths"]["outputs"]) / "contexts" / args.dataset / f"sarag__{gen}__B{b}.jsonl"
                    if not ctx_file.exists():
                        print(f"[skip] {gen} B={b}: run the pack stage first ({ctx_file.name})")
                        continue
                    rows = list(read_jsonl(ctx_file))[:args.n]
                    bench_one(llm, pb.render(qmap[rows[0]["question_id"]], rows[0]["context"]), 8)   # warm-up
                    res = [bench_one(llm, pb.render(qmap[r["question_id"]], r["context"]), args.max_tokens) for r in rows]
                    rec = {"generator": gen, "setting": setting, "budget": b, "n": len(res),
                           "gguf": gguf.name, "cpu_threads": args.cpu_threads if setting == "cpu" else None,
                           **{k: float(np.mean([x[k] for x in res])) for k in ("ttft_s", "tok_per_s", "latency_s", "new_tokens")},
                           "peak_rss_mib": mem.rss / 2**20, "peak_gpu_mib": mem.gpu}
                    with open(out_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                del llm


if __name__ == "__main__":
    main()
