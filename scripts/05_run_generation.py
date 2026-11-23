"""Expand, pack to the token budget and generate cited answers (resumable JSONL).

Roadmap: Generation
"""

from sarag.utils.cli import base_parser, setup


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = setup(args)
    print(f"[{cfg.get('name', 'default')}] 05_run_generation: not implemented yet")
    raise NotImplementedError


if __name__ == "__main__":
    main()
