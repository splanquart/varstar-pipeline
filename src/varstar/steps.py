"""Pipeline steps (only `inventory` is implemented at milestone M1)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .discovery import discover
from .domain.model import Inventory
from .inventory import build_inventory
from .inventory_io import inventory_markdown, inventory_to_dict
from .provenance import (code_fingerprint, config_fingerprint, files_fingerprint, read_manifest, update_provenance,
                         write_manifest)

STEPS = ["inventory", "check", "masters", "calibrate", "astrometry", "quality", "catalog", "photometry",
         "differential", "detect", "report"]

log = logging.getLogger("varstar")


def setup_logging(workdir: Path, step: str, level: str = "INFO") -> None:
    (workdir / "logs").mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("varstar")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    fh = logging.FileHandler(workdir / "logs" / f"{step}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)


@dataclass
class InventoryResult:
    inventory: Inventory | None
    data: dict[str, Any]
    skipped: bool
    workdir: Path


def run_inventory(source: Path, cfg: dict[str, Any], workdir: Path, force: bool = False) -> InventoryResult:
    source = source.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, "inventory", cfg.get("runtime", {}).get("log_level", "INFO"))

    files = discover(source, cfg, exclude=[workdir]).fits_files
    fingerprint = {
        "files": files_fingerprint(files),
        "config": config_fingerprint(cfg, ["inventory", "matching", "qa"]),
        "version": __version__,
        "code": code_fingerprint(),
    }
    previous = read_manifest(workdir, "inventory")
    json_path = workdir / "inventory.json"
    if not force and previous and previous.get("fingerprint") == fingerprint and json_path.exists():
        log.info("Inventaire inchangé, étape sautée")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return InventoryResult(None, data, True, workdir)

    log.info("Inventaire de %s (%d fichiers FITS)", source, len(files))
    inv = build_inventory(source, cfg, exclude=[workdir])
    data = inventory_to_dict(inv)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    from .qa import header_checks, items_markdown
    qa_items = header_checks(inv, cfg)
    (workdir / "inventory.md").write_text(
        inventory_markdown(inv, items_markdown(qa_items, "## Contrôles qualité (en-têtes)")), encoding="utf-8")
    (workdir / "config.used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                              encoding="utf-8")
    for a in inv.anomalies:
        getattr(log, {"error": "error", "warning": "warning"}.get(a.severity, "info"))(a.message)
    write_manifest(workdir, "inventory", {"fingerprint": fingerprint, "n_files": len(files)})
    update_provenance(workdir, "inventory", cfg, {
        "source": str(source), "n_files": len(files), "fingerprint": fingerprint,
        "outputs": ["inventory.json", "inventory.md"],
    })
    return InventoryResult(inv, data, False, workdir)
