"""Benchmark GGUF Q4_K_M generators with llama.cpp: TTFT, tok/s, latency, peak memory.

Roadmap: Local benchmarks
"""

from sarag.utils.cli import base_parser, setup


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = setup(args)
    print(f"[{cfg.get('name', 'default')}] 07_benchmark_local: not implemented yet")
    raise NotImplementedError


if __name__ == "__main__":
    main()
