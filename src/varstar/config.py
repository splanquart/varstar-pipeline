"""Configuration loading: packaged defaults deep-merged with an optional user file."""

from __future__ import annotations

import copy
import datetime as dt
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


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


def optical_setup_periods(cfg: dict[str, Any]) -> list[tuple[dt.date, dt.date]]:
    """Parse matching.same_optical_setup into a list of inclusive (start, end) dates."""
    raw = cfg.get("matching", {}).get("same_optical_setup") or []
    periods = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"same_optical_setup : période invalide {item!r} (attendu [début, fin])")
        start, end = (_as_date(x) for x in item)
        if end < start:
            start, end = end, start
        periods.append((start, end))
    return periods


def _as_date(x: Any) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def resolve_workdir(source: Path, cfg: dict[str, Any], workdir: str | Path | None = None) -> Path:
    """Default workdir: <source>/../varstar_work/<source name>."""
    chosen = workdir or cfg.get("paths", {}).get("workdir")
    if chosen:
        return Path(chosen).expanduser().resolve()
    source = source.resolve()
    return source.parent / "varstar_work" / source.name
