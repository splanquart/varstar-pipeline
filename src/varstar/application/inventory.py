"""Use case « inventaire » : read the headers, measure the ADU scale, match calibration, publish.

No file access here: everything goes through the ports (``application.ports``).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..domain.inventory import adu_scale_samples, build_inventory, usable_frames
from ..domain.model import AduScaleReading, Inventory, SensorKey
from ..domain.qa.headers import header_checks
from .ports import ArtifactStore, FrameSource, InventoryReport, PixelReader, ProvenanceRecorder

log = logging.getLogger("varstar")

FINGERPRINT_SECTIONS = ("inventory", "matching", "qa")


@dataclass
class InventoryResult:
    inventory: Inventory | None   # None when the step was skipped (outputs unchanged)
    data: dict[str, Any]          # content of inventory.json
    skipped: bool
    workdir: Path


def load_inventory(source: FrameSource, pixels: PixelReader, root: Path, cfg: dict[str, Any],
                   exclude: list[Path] | None = None, now: dt.datetime | None = None) -> Inventory:
    """Build the inventory of ``root``; the ADU scale is measured on one frame per sensor key."""
    scan = source.scan(root, cfg, exclude)
    readings: dict[SensorKey, AduScaleReading] | None = None
    if bool(cfg["inventory"].get("read_pixels_for_units", True)):
        readings = {}
        for key, frame in adu_scale_samples(usable_frames(scan.frames)).items():
            try:
                readings[key] = AduScaleReading(frame.rel_path, scale=pixels.adu_scale(frame))
            except Exception as exc:  # unreadable pixels must not stop the inventory
                readings[key] = AduScaleReading(frame.rel_path, error=str(exc))
    return build_inventory(scan, cfg, readings, now or dt.datetime.now().replace(microsecond=0))


def run_inventory(root: Path, cfg: dict[str, Any], *, store: ArtifactStore, provenance: ProvenanceRecorder,
                  source: FrameSource, pixels: PixelReader, report: InventoryReport,
                  force: bool = False) -> InventoryResult:
    """Step « inventory »: skipped when files, configuration and code are unchanged."""
    root = root.resolve()
    files = source.files(root, cfg, exclude=[store.root])
    fingerprint = {
        "files": provenance.files_fingerprint(files),
        "config": provenance.config_fingerprint(cfg, list(FINGERPRINT_SECTIONS)),
        "version": __version__,
        "code": provenance.code_fingerprint(),
    }
    previous = provenance.read_manifest("inventory")
    if not force and previous and previous.get("fingerprint") == fingerprint and store.exists("inventory.json"):
        log.info("Inventaire inchangé, étape sautée")
        return InventoryResult(None, store.read_json("inventory.json"), True, store.root)

    log.info("Inventaire de %s (%d fichiers FITS)", root, len(files))
    inv = load_inventory(source, pixels, root, cfg, exclude=[store.root])
    data = report.to_dict(inv)
    store.write_json("inventory.json", data)
    store.write_text("inventory.md", report.markdown(inv, header_checks(inv, cfg)))
    store.write_yaml("config.used.yaml", cfg)
    for a in inv.anomalies:
        getattr(log, {"error": "error", "warning": "warning"}.get(a.severity, "info"))(a.message)
    provenance.write_manifest("inventory", {"fingerprint": fingerprint, "n_files": len(files)})
    provenance.record("inventory", cfg, {
        "source": str(root), "n_files": len(files), "fingerprint": fingerprint,
        "outputs": ["inventory.json", "inventory.md"],
    })
    return InventoryResult(inv, data, False, store.root)
