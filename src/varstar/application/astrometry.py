"""Use case « astrometry » (M3): one WCS per calibrated image, segments, A* checks.

The calibrated images are never modified: each solution is stored beside them, in
astrometry/<group>.json (FITS WCS keywords included). A group whose calibration, settings,
catalogue and solver version are unchanged is skipped (manifest per group).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from ..domain.astrometry import (ASTROMETRY_VERSION, SolveHint, SolveJob, SolveResult, SolveSettings,
                                 assign_segments, catalog_cone, circular_median_deg, expected_scale_arcsec)
from ..domain.calibration import group_key
from ..domain.model import Frame, Inventory
from ..domain.plan import MastersPlan
from ..domain.qa.astrometry import astrometry_checks, field_center
from ..domain.qa.model import QAItem
from ..domain.settings import Settings
from .calibrate import manifest_rel as calibrate_manifest_rel
from .check import load_report, log_items, save_report
from .ports import (ArtifactStore, AstrometryReport, PlateSolver, Progress, ProvenanceRecorder, QAReport,
                    StarCatalog)

log = logging.getLogger("varstar")


def _silent(_: str) -> None:
    pass


def manifest_rel(key: str) -> str:
    return f"manifests/astrometry/{key}.json"


def solutions_rel(key: str) -> str:
    return f"astrometry/{key}.json"


def build_job(path: str, frame: Frame | None, cfg: Settings) -> SolveJob:
    """Solve job of one calibrated image, from the header of its raw light (no blind solving)."""
    if frame is None:
        return SolveJob(path, None, None, "en-tête de l'image brute introuvable")
    when = frame.date_obs.isoformat() if frame.date_obs else None
    if frame.ra is None or frame.dec is None:
        return SolveJob(path, None, when, "RA/DEC absents de l'en-tête (monture non connectée ?)")
    scale = cfg.astrometry.pixel_scale_arcsec or expected_scale_arcsec(frame.focallen, frame.xpixsz)
    if not scale:
        return SolveJob(path, None, when, "FOCALLEN ou XPIXSZ absent de l'en-tête (renseigner "
                                          "astrometry.pixel_scale_arcsec)")
    return SolveJob(path, SolveHint(frame.ra, frame.dec, scale, frame.naxis1, frame.naxis2), when)


def astrometry_fingerprint(calibration_fp: str, cfg: Settings, catalog_desc: str, solver: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"v": ASTROMETRY_VERSION, "calibration": calibration_fp, "solver": solver,
                         "catalog": catalog_desc, "settings": cfg.section("astrometry")},
                        sort_keys=True, default=str).encode())
    return h.hexdigest()


def _image_row(raw: str, cal: str, job: SolveJob, res: SolveResult) -> dict[str, Any]:
    row: dict[str, Any] = {"raw": raw, "calibrated": cal, "time": job.time, **res.to_dict()}
    if res.ok and res.solution is not None and job.hint is not None:
        row["center"] = list(field_center(res.solution, job.hint.width, job.hint.height))
        row["field_deg"] = list(job.hint.field_deg)
    return row


def _group_row(key: str, label: str, images: Iterable[dict[str, Any]]) -> dict[str, Any]:
    images = list(images)
    ok = [r for r in images if r["ok"]]

    def med(field: str) -> float | None:
        vals = [r[field] for r in ok if r.get(field) is not None]
        return float(np.median(vals)) if vals else None

    return {"group": key, "label": label, "n": len(images), "n_solved": len(ok),
            "rms_median_arcsec": med("rms_arcsec"), "scale_median_arcsec": med("scale_arcsec"),
            "orientation_median_deg": circular_median_deg(r["orientation_deg"] for r in ok) if ok else None,
            "segments": sorted({r["segment"] for r in ok if r.get("segment") is not None})}


def run_astrometry(plan: MastersPlan, keys: list[str], cfg: Settings, inv: Inventory, *, solver: PlateSolver,
                   catalog: StarCatalog, store: ArtifactStore, provenance: ProvenanceRecorder, qa_report: QAReport,
                   report: AstrometryReport, saturation_adu: float | None = None,
                   progress: Progress = _silent, force: bool = False,
                   now: dt.datetime | None = None) -> dict[str, Any]:
    """Step « astrometry » for the calibrated groups ``keys``; returns per-group solutions and errors."""
    a = cfg.astrometry
    settings = SolveSettings.of(a, saturation_adu)
    frames = {str(f.path): f for f in inv.frames}
    errors: dict[str, str] = {}
    todo: dict[str, dict[str, Any]] = {}          # key → {calibration, jobs, raw_of}
    for key in keys:
        group = plan.groups.get(key)
        cal = store.read_json(calibrate_manifest_rel(key))
        if group is None or cal is None:
            errors[key] = "images calibrées absentes (relancer l'étape « calibrate »)"
            continue
        outputs: dict[str, str] = cal["outputs"]
        jobs = [build_job(out, frames.get(raw), cfg) for raw, out in outputs.items()]
        todo[key] = {"group": group, "calibration": cal["fingerprint"], "jobs": jobs,
                     "raw_of": {out: raw for raw, out in outputs.items()}}

    results: dict[str, dict[str, Any]] = {}
    hints = [j.hint for t in todo.values() for j in t["jobs"] if j.hint is not None]
    stars = None
    if hints:
        ra, dec, radius = catalog_cone(hints, a.pointing_tolerance_deg)
        desc = catalog.describe()
        pending = []
        for key, t in todo.items():
            fp = astrometry_fingerprint(t["calibration"], cfg, desc, solver.name())
            prev = None if force else store.read_json(manifest_rel(key))
            stored = store.read_json(solutions_rel(key))
            if prev and prev.get("fingerprint") == fp and stored:
                log.info("Astrométrie %s inchangée, sautée", key)
                results[key] = {**stored, "skipped": True}
            else:
                pending.append((key, fp))
        if pending:
            progress(f"  catalogue : {desc}, cône de {radius:g}° autour de RA {ra:.3f}°, Dec {dec:+.3f}° "
                     f"(G < {a.catalog_mag_limit:g}) …")
            stars = catalog.cone(ra, dec, radius, a.catalog_mag_limit)
            log.info("Catalogue %s : %d étoiles", stars.source, stars.n)
            n_img = sum(len(todo[k]["jobs"]) for k, _ in pending)
            progress(f"  astrométrie de {n_img} image(s) ({len(pending)} groupe(s), {stars.n} étoiles de référence) …")
            solved = solver.solve([todo[k]["jobs"] for k, _ in pending], stars, settings, progress)
            for key, fp in pending:
                t = todo[key]
                images = {t["raw_of"][j.path]: _image_row(t["raw_of"][j.path], j.path, j, solved[j.path])
                          for j in t["jobs"]}
                results[key] = {"group": key, "fingerprint": fp, "catalog": stars.source, "n_catalog": stars.n,
                                "solver": solver.name(), "images": images, "skipped": False}
    for key, t in todo.items():
        if key not in results:   # no usable hint at all in the group
            images = {t["raw_of"][j.path]: _image_row(t["raw_of"][j.path], j.path, j,
                                                      SolveResult(False, reason=j.problem)) for j in t["jobs"]}
            results[key] = {"group": key, "fingerprint": None, "catalog": None, "n_catalog": 0,
                            "solver": solver.name(), "images": images, "skipped": False}

    # Groups solved by an earlier run (other sessions or filters) take part in the segments and the
    # A* checks, so that a run on one session does not hide the findings of the others.
    every = dict(results)
    for key in plan.groups:
        if key not in every and (stored := store.read_json(solutions_rel(key))):
            every[key] = {**stored, "skipped": True}

    # Segments: per night, over every filter (a meridian flip turns the whole set-up).
    by_session: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for key, res in every.items():
        for raw, row in res["images"].items():
            if row["ok"]:
                by_session[plan.groups[key].session].append((key, raw, row))
    for session, rows in by_session.items():
        rows.sort(key=lambda kr: (kr[2]["time"] or "", kr[1]))
        seg = assign_segments([((k, raw), r["orientation_deg"]) for k, raw, r in rows], a.segment_angle_deg)
        for k, raw, r in rows:
            r["segment"] = seg[(k, raw)]
            r["segment_id"] = f"{session}/{seg[(k, raw)]}"
    for key, res in every.items():
        for row in res["images"].values():
            row.setdefault("segment", None)
            row.setdefault("segment_id", None)
        # Rewritten even when skipped: segments may change when another group of the night is added.
        store.write_json(solutions_rel(key), {k: v for k, v in res.items() if k != "skipped"})
        if res.get("fingerprint"):
            store.write_json(manifest_rel(key), {"fingerprint": res["fingerprint"]})

    # A* checks over every group of the run, recorded in qa.json (stage « astrometry »).
    borrowed = {group_key(lg): bool(lg.flat and (lg.flat.level != "A" or lg.flat.reasons))
                for lg in inv.light_groups}
    groups_qa = []
    for key, res in every.items():
        g = plan.groups[key]
        first = next((frames.get(r) for r in g.files if frames.get(r)), None)
        groups_qa.append({
            "key": key, "session": g.session, "label": g.label, "filter": g.filter,
            "flat_id": g.flat.id if g.flat else None, "flat_borrowed": borrowed.get(key, False),
            "expected_scale": a.pixel_scale_arcsec or (expected_scale_arcsec(first.focallen, first.xpixsz)
                                                       if first else None),
            "focallen": first.focallen if first else None, "xpixsz": first.xpixsz if first else None,
            "images": list(res["images"].values())})
    items = astrometry_checks(groups_qa, a, cfg.qa.pointing_max_field_frac)
    rows = [_group_row(key, plan.groups[key].display, res["images"].values()) for key, res in every.items()]
    _record(store, qa_report, items, rows, now)
    log_items(items)

    summary = {"catalog": stars.source if stars is not None else next(
        (r["catalog"] for r in results.values() if r.get("catalog")), None),
        "solver": solver.name(), "groups": results, "rows": rows, "errors": errors,
        "n_images": sum(len(r["images"]) for r in results.values()),
        "n_solved": sum(1 for r in results.values() for row in r["images"].values() if row["ok"])}
    store.write_text("astrometry/astrometry.md", report.markdown(summary, items))
    provenance.record("astrometry", cfg, {
        "solver": solver.name(), "catalog": summary["catalog"],
        "images": {row["calibrated"]: {"ok": row["ok"], "rms_arcsec": row.get("rms_arcsec"),
                                       "segment": row.get("segment_id"), "reason": row.get("reason")}
                   for r in results.values() for row in r["images"].values()},
        "errors": errors})
    return {**summary, "qa": [i.to_dict() for i in items]}


def _record(store: ArtifactStore, renderer: QAReport, items: list[QAItem], rows: list[dict[str, Any]],
            now: dt.datetime | None) -> None:
    """Replace the astrometry findings and rows in qa.json (they cover every solved group)."""
    rep = load_report(store)
    rep["astrometry"] = [i.to_dict() for i in items]
    rep.setdefault("tables", {})["astrometry"] = sorted(rows, key=lambda r: r["group"])
    save_report(store, rep, renderer, now)


def load_solutions(store: ArtifactStore, key: str) -> dict[str, Any] | None:
    """Solutions of a group (astrometry/<key>.json), for the next steps."""
    return store.read_json(solutions_rel(key))
