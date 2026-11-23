"""Build Table 1, Table 2 and Figure 1 from outputs/metrics.

Roadmap: Tables and figures
"""

from sarag.utils.cli import base_parser, setup


def main() -> None:
    args = base_parser(__doc__.splitlines()[0]).parse_args()
    cfg = setup(args)
    print(f"[{cfg.get('name', 'default')}] 08_make_tables_figures: not implemented yet")
    raise NotImplementedError


if __name__ == "__main__":
    main()
