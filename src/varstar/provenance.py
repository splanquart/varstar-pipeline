"""Run provenance (provenance.json) and step manifests for idempotence."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

from . import __version__

_PACKAGES = ("numpy", "astropy", "photutils", "astroquery", "pandas", "matplotlib", "pyyaml", "typer")


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "varstar": __version__, "platform": platform.platform()}
    for pkg in _PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def files_fingerprint(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def code_fingerprint() -> str:
    """Hash of the varstar sources, so that a code change invalidates cached steps."""
    h = hashlib.sha256()
    pkg = Path(__file__).parent
    for p in sorted(pkg.rglob("*")):
        if p.suffix in (".py", ".yaml"):
            h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def config_fingerprint(cfg: dict[str, Any], sections: Iterable[str]) -> str:
    sub = {s: cfg.get(s) for s in sections}
    return hashlib.sha256(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()


def read_manifest(workdir: Path, step: str) -> dict[str, Any] | None:
    path = workdir / "manifests" / f"{step}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_manifest(workdir: Path, step: str, data: dict[str, Any]) -> None:
    path = workdir / "manifests" / f"{step}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def update_provenance(workdir: Path, step: str, cfg: dict[str, Any], details: dict[str, Any]) -> None:
    path = workdir / "provenance.json"
    prov: dict[str, Any] = {}
    if path.exists():
        try:
            prov = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prov = {}
    prov["software"] = software_versions()
    prov["config"] = cfg
    prov.setdefault("steps", {})[step] = {"finished": dt.datetime.now().isoformat(timespec="seconds"),
                                          **details}
    path.write_text(json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
