"""Use case « calibrate » : calibrate each light group with its masters, then the C* checks.

One group failing never stops the others; a group whose masters, masks and raw files are
unchanged is skipped (manifest per group in manifests/calibrate/).
"""

from __future__ import annotations

import logging
from pathlib import PurePath
from typing import Any

from ..domain.calibration import (CalibrationSensor, calibrated_name, calibrated_subdir,
                                  calibration_fingerprint)
from ..domain.plan import GroupPlan, MastersPlan
from ..domain.settings import Settings
from .check import calibrated_checks, load_report, record_calibrated_checks
from .ports import ArtifactStore, Calibrator, FrameSource, PixelStats, Progress, ProvenanceRecorder, QAReport

log = logging.getLogger("varstar")


def _silent(_: str) -> None:
    pass


def manifest_rel(key: str) -> str:
    return f"manifests/calibrate/{key}.json"


def _rel(store: ArtifactStore, path: str) -> str | None:
    try:
        return str(PurePath(path).relative_to(store.root))
    except ValueError:
        return None


def calibrate_group(group: GroupPlan, *, calibrator: Calibrator, source: FrameSource, store: ArtifactStore,
                    sensor: CalibrationSensor, force: bool = False) -> dict[str, Any]:
    """Calibrate one light group; returns {key, fingerprint, outputs (raw → calibrated), level, skipped}."""
    fp = calibration_fingerprint(group.to_dict(), [(raw, source.file_stat(raw)) for raw in group.files])
    prev = None if force else store.read_json(manifest_rel(group.key))
    if prev and prev.get("fingerprint") == fp and all(
            (rel := _rel(store, p)) is not None and store.exists(rel) for p in prev["outputs"].values()):
        log.info("Calibration %s inchangée, sautée", group.key)
        return {**prev, "skipped": True}

    subdir = calibrated_subdir(group.session, group.label, group.filter)
    outputs = {raw: store.root / subdir / calibrated_name(raw) for raw in group.files}
    calibrator.calibrate(group, outputs, sensor)
    result = {"key": group.key, "fingerprint": fp, "outputs": {raw: str(p) for raw, p in outputs.items()},
              "level": group.level}
    store.write_json(manifest_rel(group.key), result)
    log.info("Calibration %s : %d images", group.key, len(outputs))
    return {**result, "skipped": False}


def run_calibrate(plan: MastersPlan, keys: list[str], cfg: Settings, sensor: CalibrationSensor, *,
                  calibrator: Calibrator, source: FrameSource, store: ArtifactStore, pixels: PixelStats,
                  qa_report: QAReport, provenance: ProvenanceRecorder,
                  progress: Progress = _silent, force: bool = False) -> dict[str, Any]:
    """Step « calibrate » for the groups ``keys`` of the plan; returns results, errors and C* findings."""
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for key in keys:
        group = plan.groups.get(key)
        if group is None:
            errors[key] = "groupe absent du plan des masters (relancer l'étape masters)"
            continue
        progress(f"  calibration {group.display} ({len(group.files)} images) …")
        try:
            results[key] = calibrate_group(group, calibrator=calibrator, source=source, store=store, sensor=sensor,
                                           force=force)
        except Exception as exc:  # one failing group must not stop the others
            log.error("Calibration %s en échec : %s", key, exc)
            errors[key] = str(exc)
    if results:
        groups = {k: plan.groups[k].to_dict() for k in results}
        items, rows = calibrated_checks({"groups": groups}, results, cfg, pixels=pixels)
        record_calibrated_checks(store, qa_report, items, rows, results)
        for i in items:
            getattr(log, "error" if i.severity == "error" else "warning")(f"[{i.check}] {i.message}")
    provenance.record("calibrate", cfg, {
        "images": {raw: out for r in results.values() for raw, out in r["outputs"].items()},
        "groups": {k: {f: groups[k][f] for f in ("level", "dark", "flat", "darkflat", "bpm", "ampglow")}
                   for k in results} if results else {},
        "errors": errors,
        "siril_python_init_guard": calibrator.status(),
    })
    return {"results": results, "errors": errors, "qa": load_report(store).get("calibrated", [])}

