"""Parse the law corpus into document trees (C1); report parse success / fallback rate.

Roadmap: Parser
"""

from sarag.utils.cli import base_parser, setup


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = setup(args)
    print(f"[{cfg.get('name', 'default')}] 02_parse_corpus: not implemented yet")
    raise NotImplementedError


if __name__ == "__main__":
    main()
