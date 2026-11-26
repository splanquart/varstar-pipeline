"""Use case « contrôles qualité » (docs/QA_CHECKS.md): header, raw pixel and post-calibration checks.

The rules live in ``domain.qa``; the pixels are measured through the ``PixelStats`` port and the
report (qa/qa.json + qa/qa.md) is published through the ``ArtifactStore``.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable

import numpy as np

from ..domain.model import FLAT, Inventory
from ..domain.qa.calibrated import ampglow_residual_items, calibrated_group_checks
from ..domain.qa.headers import header_checks
from ..domain.qa.model import QAItem, sort_items
from ..domain.qa.pixels import (ampglow_zones, calibrated_frame_stats, flat_compare_bin, flat_compare_frames,
                                flat_ratio_metrics)
from ..domain.qa.raw import (flat_bin_adapted, flat_compare_failed, flat_comparison_series, flat_dark_level,
                             flat_pair_check, flat_pair_skipped, frames_to_measure, raw_pixel_checks, saturation_adu)
from ..domain.settings import Settings
from .ports import ArtifactStore, PixelStats, Progress, QAReport

log = logging.getLogger("varstar")

REPORT_JSON = "qa/qa.json"
REPORT_MD = "qa/qa.md"


def _silent(_: str) -> None:
    pass


def load_report(store: ArtifactStore) -> dict[str, Any]:
    """The current qa.json (empty when absent or unreadable)."""
    try:
        data = store.read_json(REPORT_JSON)
    except ValueError:   # corrupted JSON: start again
        return {}
    return data if isinstance(data, dict) else {}


def save_report(store: ArtifactStore, report: dict[str, Any], renderer: QAReport,
                now: dt.datetime | None = None) -> None:
    report["updated"] = (now or dt.datetime.now()).isoformat(timespec="seconds")
    store.write_json(REPORT_JSON, report)
    store.write_text(REPORT_MD, renderer.markdown(report))


def log_items(items: Iterable[dict[str, Any]] | Iterable[QAItem]) -> None:
    for i in items:
        d = i.to_dict() if isinstance(i, QAItem) else i
        getattr(log, {"error": "error", "warning": "warning"}.get(d["severity"], "info"))(
            f"[{d['check']}] {d['message']}")


def raw_checks(inv: Inventory, cfg: Settings, *, store: ArtifactStore, pixels: PixelStats,
               progress: Progress = _silent) -> tuple[list[QAItem], dict[str, Any]]:
    """P1–P8 on the raw frames; the flat ratios of P4 are written to qa/flats/."""
    q = cfg.qa
    stats = pixels.frame_stats(frames_to_measure(inv), saturation_adu(inv), progress="contrôles")
    res = raw_pixel_checks(inv, cfg, stats)
    items, tables = list(res.items), dict(res.tables)
    if q.flat_compare and inv.calibration_sets.get(FLAT):
        max_frames = q.flat_compare_max_frames
        rows = []
        for filt, sets in flat_comparison_series(inv):
            progress(f"  contrôles : comparaison des flats {filt} ({len(sets)} séries) …")
            # Binning reduced on small images to keep enough blocks, and said so:
            # https://github.com/splanquart/varstar-pipeline/issues/4
            shape = (sets[0].frames[0].naxis2, sets[0].frames[0].naxis1)
            bin_px = flat_compare_bin(shape, q.flat_compare_bin)
            if bin_px < q.flat_compare_bin:
                items.append(flat_bin_adapted(filt, q.flat_compare_bin, bin_px, shape))
            # A failing check is reported (P4 error), it does not stop the run (https://github.com/splanquart/varstar-pipeline/issues/2).
            try:
                flats = {s.id: pixels.quick_flat(s.id, flat_compare_frames(s.frames, max_frames), bin_px,
                                                 flat_dark_level(s, inv, cfg, res.summaries)) for s in sets}
            except Exception as exc:
                log.exception("comparaison des flats %s", filt)
                items.append(flat_compare_failed(filt, sets, exc))
                continue
            for a, b in zip(sets, sets[1:]):
                try:
                    met = flat_ratio_metrics(flats[a.id], flats[b.id])
                except Exception as exc:
                    log.exception("comparaison des flats %s / %s", a.id, b.id)
                    items.append(flat_compare_failed(filt, [a, b], exc))
                    continue
                if met is None:
                    items.append(flat_pair_skipped(filt, a, b, flats[a.id].shape, bin_px))
                    continue
                rel = f"qa/flats/ratio_{a.id}_vs_{b.id}.fits"
                store.write_image(rel, (flats[a.id] / flats[b.id]).astype(np.float32))
                item, row = flat_pair_check(inv, cfg, filt, a, b, met, str(store.root / rel))
                row["bin_px"] = bin_px
                rows.append(row)
                if item is not None:
                    items.append(item)
        tables["flat_pairs"] = rows
    return sort_items(items), tables


def run_checks(inv: Inventory, cfg: Settings, *, store: ArtifactStore, pixels: PixelStats,
               report: QAReport, raw: bool = True, progress: Progress = _silent,
               now: dt.datetime | None = None) -> dict[str, Any]:
    """Header (+ raw pixel) checks; publishes qa/qa.json and qa/qa.md. Calibrated checks are kept."""
    rep = load_report(store)
    rep["header"] = [i.to_dict() for i in header_checks(inv, cfg)]
    if raw:
        items, tables = raw_checks(inv, cfg, store=store, pixels=pixels, progress=progress)
        rep["raw"] = [i.to_dict() for i in items]
        rep.setdefault("tables", {}).update(tables)
    save_report(store, rep, report, now)
    log_items(rep["header"] + rep.get("raw", []))
    return rep


def calibrated_checks(plan: dict[str, Any], results: dict[str, Any], cfg: Settings, *,
                      pixels: PixelStats, step: int = 4) -> tuple[list[QAItem], list[dict[str, Any]]]:
    """C1 and C3 on the calibrated frames of each group (``results``: group key → calibration result)."""
    items: list[QAItem] = []
    rows: list[dict[str, Any]] = []
    for key, res in results.items():
        group = plan["groups"][key]
        zones = []
        if group.get("ampglow"):
            try:
                mask = pixels.read_image(group["ampglow"])
            except FileNotFoundError:
                mask = None
            if mask is not None:
                zones = ampglow_zones(mask[::step, ::step] > 0, step)
        per = [{"file": out, "raw": raw, **calibrated_frame_stats(pixels.read_image(out)[::step, ::step], zones)}
               for raw, out in res["outputs"].items()]
        group_items, row = calibrated_group_checks(key, group, per, cfg)
        if row is None:
            continue
        rows.append(row)
        items += group_items
    items += ampglow_residual_items(rows)
    return sort_items(items), rows


def record_calibrated_checks(store: ArtifactStore, renderer: QAReport, items: list[QAItem],
                             rows: list[dict[str, Any]], groups: Iterable[str],
                             now: dt.datetime | None = None) -> dict[str, Any]:
    """Replace the calibrated findings and rows of ``groups`` in the report; other groups are kept."""
    groups = set(groups)
    rep = load_report(store)
    old = {r["group"]: r for r in rep.get("tables", {}).get("calibrated", [])}
    old.update({r["group"]: r for r in rows})
    keep = [d for d in rep.get("calibrated", []) if d.get("data", {}).get("group") not in groups]
    rep["calibrated"] = keep + [i.to_dict() for i in items]
    rep.setdefault("tables", {})["calibrated"] = list(old.values())
    save_report(store, rep, renderer, now)
    return rep
