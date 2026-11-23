"""Build index units (clause/article/fixed) with breadcrumbs, then BM25 and bge-m3 indexes.

Roadmap: Indexes
"""

from sarag.utils.cli import base_parser, setup


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = setup(args)
    print(f"[{cfg.get('name', 'default')}] 03_build_index: not implemented yet")
    raise NotImplementedError


if __name__ == "__main__":
    main()
