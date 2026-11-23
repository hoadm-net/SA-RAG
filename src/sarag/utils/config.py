"""YAML config loading with `_base_` inheritance and dotted-key overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving `_base_` (relative to the file) recursively.

    `overrides` are strings like ``"packing.budget=1024"``; values are parsed as YAML.
    """
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path) if not path.exists() else path.resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_models(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    return load_config(path)


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
