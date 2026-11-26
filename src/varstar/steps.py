"""Pipeline steps (only `inventory` is implemented at milestone M1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .adapters.discovery import FilesystemFrameSource
from .adapters.fits_io import FitsPixelReader
from .adapters.provenance import ProvenanceStore
from .adapters.reporting.inventory_report import InventoryReportAdapter
from .adapters.workspace import Workspace
from .application import inventory as inventory_use_case
from .application.inventory import InventoryResult  # noqa: F401  (re-exported)

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


def run_inventory(source: Path, cfg: dict[str, Any], workdir: Path, force: bool = False) -> InventoryResult:
    """Wires the file adapters to the use case (until the composition root, step 6)."""
    source = source.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, "inventory", cfg.get("runtime", {}).get("log_level", "INFO"))
    return inventory_use_case.run_inventory(
        source, cfg, store=Workspace(workdir), provenance=ProvenanceStore(workdir), source=FilesystemFrameSource(),
        pixels=FitsPixelReader(), report=InventoryReportAdapter(), force=force)
