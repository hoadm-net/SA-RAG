"""Generation backends: vLLM for the main experiments (llama.cpp is used by scripts/07_benchmark_local.py)."""

from __future__ import annotations

import time


class VLLMGenerator:
    def __init__(self, model_id: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.85,
                 max_new_tokens: int = 512, seed: int = 42, **engine_kwargs):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model_id, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       seed=seed, **engine_kwargs)
        # Greedy decoding (temperature 0) for every run.
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)

    def generate(self, prompts: list[str]) -> tuple[list[str], float]:
        """Returns the completions and the wall time per prompt (batched, so latency is amortised;
        per-question latency on commodity hardware is measured separately with llama.cpp)."""
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, self.params, use_tqdm=False)
        dt = (time.perf_counter() - t0) / max(len(prompts), 1)
        return [o.outputs[0].text for o in outs], dt
