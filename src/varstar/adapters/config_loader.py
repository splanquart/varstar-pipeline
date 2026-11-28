"""Configuration files: packaged defaults ← user YAML ← overrides, checked into typed ``Settings``."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ..domain.settings import Settings, SettingsError


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


def _text_keys(value: Any) -> Any:
    """Mapping keys as text: YAML reads ``2021-09-02:`` as a date, which JSON (fingerprints,
    provenance.json) cannot use as a key. Values are left alone (dates in lists stay dates)."""
    if isinstance(value, dict):
        return {(k.isoformat() if hasattr(k, "isoformat") else k): _text_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_text_keys(v) for v in value]
    return value


def load_raw(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The merged configuration as a plain dict (defaults ← file ← overrides)."""
    cfg = load_defaults()
    if path is not None:
        try:
            with open(path, encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise SettingsError(f"Fichier de configuration illisible ({path}) : {exc}") from None
        if not isinstance(user, dict):
            raise SettingsError(f"Fichier de configuration invalide ({path}) : sections attendues")
        unknown = set(user) - set(cfg)
        if unknown:
            raise SettingsError(f"Sections de configuration inconnues : {sorted(unknown)}")
        cfg = deep_merge(cfg, _text_keys(user))
    if overrides:
        cfg = deep_merge(cfg, _text_keys(overrides))
    return cfg


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    """Effective, checked configuration (``SettingsError`` with a French message if invalid)."""
    return Settings.from_dict(load_raw(path, overrides))


def resolve_workdir(source: Path, cfg: Settings, workdir: str | Path | None = None) -> Path:
    """Default workdir: <source>/../varstar_work/<source name>."""
    chosen = workdir or cfg.paths.workdir
    if chosen:
        return Path(chosen).expanduser().resolve()
    source = source.resolve()
    return source.parent / "varstar_work" / source.name
