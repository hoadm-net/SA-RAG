"""Common argparse setup for scripts/."""

from __future__ import annotations

import argparse

from .config import load_config
from .seed import set_seed


def base_parser(description: str, default_config: str = "configs/experiments/sarag.yaml") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="experiment YAML")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. packing.budget=1024")
    return p


def setup(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 42))
    return cfg
