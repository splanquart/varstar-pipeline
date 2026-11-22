"""Configuration loading: packaged defaults deep-merged with an optional user file."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .domain.settings import optical_setup_periods  # noqa: F401  (re-exported)


def load_defaults() -> dict[str, Any]:
    text = resources.files("varstar").joinpath("defaults.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the effective configuration (defaults <- file <- overrides)."""
    cfg = load_defaults()
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        unknown = set(user) - set(cfg)
        if unknown:
            raise ValueError(f"Sections de configuration inconnues : {sorted(unknown)}")
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve_workdir(source: Path, cfg: dict[str, Any], workdir: str | Path | None = None) -> Path:
    """Default workdir: <source>/../varstar_work/<source name>."""
    chosen = workdir or cfg.get("paths", {}).get("workdir")
    if chosen:
        return Path(chosen).expanduser().resolve()
    source = source.resolve()
    return source.parent / "varstar_work" / source.name
