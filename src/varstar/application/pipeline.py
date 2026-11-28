"""Use case « run » : the steps in order, resumable with ``--from`` (steps after M3 are not implemented).

The adapters are given by the composition root (``varstar.cli.bootstrap``) as ``PipelineServices``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..domain.calibration import CalibrationSensor, PipelineError, group_key, select_groups
from ..domain.settings import Settings
from .astrometry import run_astrometry
from .calibrate import run_calibrate
from .check import run_checks
from .inventory import load_inventory, run_inventory
from .masters import load_plan, run_masters
from .ports import (ArtifactStore, AstrometryReport, Calibrator, FrameSource, InventoryReport, MasterBuilder,
                    MasterRepository, MastersReport, PixelReader, PixelStats, PlateSolver, ProvenanceRecorder,
                    QAReport, ReferenceMasters, Reporter, StarCatalog)

STEPS = ["inventory", "check", "masters", "calibrate", "astrometry", "quality", "catalog", "photometry",
         "differential", "detect", "report"]
IMPLEMENTED = ["inventory", "check", "masters", "calibrate", "astrometry"]


@dataclass
class PipelineServices:
    """The ports used by the pipeline. ``calibration_tools`` is called only when a step needs Siril."""

    store: ArtifactStore
    provenance: ProvenanceRecorder
    source: FrameSource
    pixel_reader: PixelReader
    pixel_stats: PixelStats
    inventory_report: InventoryReport
    qa_report: QAReport
    masters_report: MastersReport
    repo: MasterRepository
    references: ReferenceMasters | None
    calibration_tools: Callable[[], tuple[MasterBuilder, Calibrator]]
    reporter: Reporter
    start_log: Callable[[str], None]   # one log file per step
    astrometry_tools: Callable[[], tuple[PlateSolver, StarCatalog]] | None = None
    astrometry_report: AstrometryReport | None = None


def check_step(step: str) -> int:
    if step not in STEPS:
        raise PipelineError(f"Étape inconnue : {step} (étapes : {', '.join(STEPS)})")
    return STEPS.index(step)


def run_pipeline(root: Path, cfg: Settings, svc: PipelineServices, from_step: str = "inventory",
                 sessions: list[str] | None = None, filters: list[str] | None = None,
                 force: bool = False) -> dict[str, Any]:
    start = check_step(from_step)
    root = root.resolve()
    summary: dict[str, Any] = {"steps": []}
    say = svc.reporter.progress

    inv = None
    if start <= STEPS.index("inventory"):
        svc.start_log("inventory")
        res = run_inventory(root, cfg, store=svc.store, provenance=svc.provenance, source=svc.source,
                            pixels=svc.pixel_reader, report=svc.inventory_report, force=force)
        inv = res.inventory
        summary["steps"].append("inventory")
    if inv is None:   # step skipped (unchanged) or resumed later: headers are read again
        inv = load_inventory(svc.source, svc.pixel_reader, root, cfg, exclude=[svc.store.root])
    groups = select_groups(inv, sessions, filters)
    if not groups:
        raise PipelineError("Aucun groupe de lights ne correspond à la sélection")
    summary["groups"] = [group_key(g) for g in groups]
    if start <= STEPS.index("check"):
        summary["qa"] = run_checks(inv, cfg, store=svc.store, pixels=svc.pixel_stats, report=svc.qa_report,
                                   progress=say)
        summary["steps"].append("check")
    builder, calibrator = svc.calibration_tools()

    if start <= STEPS.index("masters"):
        svc.start_log("masters")
        plan = run_masters(inv, groups, cfg, source=svc.source, pixels=svc.pixel_stats, builder=builder,
                           repo=svc.repo, references=svc.references, store=svc.store, provenance=svc.provenance,
                           report=svc.masters_report, progress=say)
        summary["steps"].append("masters")
    else:
        plan = load_plan(svc.store)
        if plan is None:
            raise PipelineError("plan des masters absent : relancer depuis l'étape « masters »")
    summary["plan"] = plan.to_dict()

    if start <= STEPS.index("calibrate"):
        svc.start_log("calibrate")
        summary["calibrate"] = run_calibrate(plan, summary["groups"], cfg, CalibrationSensor.of(inv),
                                             calibrator=calibrator, source=svc.source, store=svc.store,
                                             pixels=svc.pixel_stats, qa_report=svc.qa_report,
                                             provenance=svc.provenance, progress=say, force=force)
        summary["steps"].append("calibrate")

    if start <= STEPS.index("astrometry"):
        if svc.astrometry_tools is None or svc.astrometry_report is None:
            raise PipelineError("astrométrie non configurée")
        failed_keys = set(summary.get("calibrate", {}).get("errors", {}))
        keys = [k for k in summary["groups"] if k not in failed_keys]
        svc.start_log("astrometry")
        solver, catalog = svc.astrometry_tools()
        sensor = inv.sensors[0] if inv.sensors else None
        saturation = 0.9 * sensor.saturation_adu if sensor and sensor.saturation_adu else None
        summary["astrometry"] = run_astrometry(plan, keys, cfg, inv, solver=solver, catalog=catalog,
                                               store=svc.store, provenance=svc.provenance,
                                               qa_report=svc.qa_report, report=svc.astrometry_report,
                                               saturation_adu=saturation, progress=say, force=force)
        summary["steps"].append("astrometry")

    summary["not_implemented"] = [s for s in STEPS[max(start, STEPS.index("astrometry") + 1):]
                                  if s not in IMPLEMENTED]
    return summary
