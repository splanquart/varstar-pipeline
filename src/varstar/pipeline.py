"""Pipeline driver: inventory -> check -> masters -> calibrate (steps after M2 are not implemented yet).

Wires the file and Siril adapters to the use cases until the composition root (step 6).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import qa
from .adapters.discovery import FilesystemFrameSource
from .adapters.library import DiskMasterRepository
from .adapters.pixel_stats import PixelStatsCache
from .adapters.pixinsight import PixInsightMasters
from .adapters.provenance import ProvenanceStore
from .adapters.reporting.masters_markdown import MastersReportAdapter, masters_markdown  # noqa: F401
from .adapters.reporting.qa_markdown import QAReportAdapter
from .adapters.siril import Siril
from .adapters.siril_calibrator import SirilCalibrator
from .adapters.siril_masters import SirilMasterBuilder
from .adapters.workspace import Workspace
from .application import calibrate as calibrate_use_case
from .application import masters as masters_use_case
from .domain.calibration import CalibrationSensor, PipelineError, group_key, select_groups  # noqa: F401
from .domain.model import Inventory, LightGroup
from .domain.plan import MastersPlan
from .inventory import build_inventory
from .steps import STEPS, run_inventory, setup_logging

log = logging.getLogger("varstar")

IMPLEMENTED = ["inventory", "check", "masters", "calibrate"]


def _say(message: str) -> None:
    print(message, flush=True)


def run_masters(inv: Inventory, groups: list[LightGroup], cfg: dict[str, Any], workdir: Path,
                siril: Siril) -> dict[str, Any]:
    setup_logging(workdir, "masters", cfg["runtime"].get("log_level", "INFO"))
    library = DiskMasterRepository(Path(cfg["paths"]["library"]).expanduser())
    plan = masters_use_case.run_masters(
        inv, groups, cfg, source=FilesystemFrameSource(), pixels=PixelStatsCache.from_config(workdir, cfg),
        builder=SirilMasterBuilder(siril, library, workdir / "siril", cfg), repo=library,
        references=PixInsightMasters.from_config(cfg, inv.root), store=Workspace(workdir),
        provenance=ProvenanceStore(workdir), report=MastersReportAdapter(), progress=_say)
    return plan.to_dict()


def run_calibrate(plan: dict[str, Any], inv: Inventory, keys: list[str], cfg: dict[str, Any], workdir: Path,
                  siril: Siril, force: bool = False) -> dict[str, Any]:
    setup_logging(workdir, "calibrate", cfg["runtime"].get("log_level", "INFO"))
    return calibrate_use_case.run_calibrate(
        MastersPlan.from_dict(plan), keys, cfg, CalibrationSensor.of(inv), calibrator=SirilCalibrator(siril, workdir),
        source=FilesystemFrameSource(), store=Workspace(workdir), pixels=PixelStatsCache(workdir),
        qa_report=QAReportAdapter(), provenance=ProvenanceStore(workdir), tool_status=siril.guard_status(),
        progress=_say, force=force)


# --------------------------------------------------------------------------- driver


def run_pipeline(source: Path, cfg: dict[str, Any], workdir: Path, from_step: str = "inventory",
                 sessions: list[str] | None = None, filters: list[str] | None = None,
                 force: bool = False) -> dict[str, Any]:
    if from_step not in STEPS:
        raise PipelineError(f"Étape inconnue : {from_step} (étapes : {', '.join(STEPS)})")
    start = STEPS.index(from_step)
    summary: dict[str, Any] = {"steps": []}

    if start <= STEPS.index("inventory"):
        run_inventory(source, cfg, workdir, force=force)
        summary["steps"].append("inventory")
    inv = build_inventory(source, cfg, exclude=[workdir])
    groups = select_groups(inv, sessions, filters)
    if not groups:
        raise PipelineError("Aucun groupe de lights ne correspond à la sélection")
    summary["groups"] = [group_key(g) for g in groups]
    if start <= STEPS.index("check"):
        summary["qa"] = qa.run_checks(inv, cfg, workdir)
        summary["steps"].append("check")
    siril = Siril.from_config(cfg, workdir)

    plan_path = workdir / "masters" / "plan.json"
    if start <= STEPS.index("masters"):
        plan = run_masters(inv, groups, cfg, workdir, siril)
        summary["steps"].append("masters")
    else:
        if not plan_path.exists():
            raise PipelineError("plan des masters absent : relancer depuis l'étape « masters »")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary["plan"] = plan

    if start <= STEPS.index("calibrate"):
        summary["calibrate"] = run_calibrate(plan, inv, summary["groups"], cfg, workdir, siril, force=force)
        summary["steps"].append("calibrate")

    remaining = [s for s in STEPS[max(start, STEPS.index("calibrate") + 1):] if s not in IMPLEMENTED]
    summary["not_implemented"] = remaining
    return summary
