"""JSON-ready dicts of the domain objects (inventory.json)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from ..domain.inventory import darks_todo
from ..domain.model import DarkTodo, FrameSet, Inventory, LightGroup, Match


def _iso(x: dt.datetime | dt.date | None) -> str | None:
    return x.isoformat() if x is not None else None


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def _set_dict(s: FrameSet) -> dict[str, Any]:
    return {
        "id": s.id, "kind": s.kind, "sensor_key": s.sensor_key._asdict(), "exptime": s.exptime,
        "filter": s.filter, "n": s.n, "start_utc": _iso(s.start), "end_utc": _iso(s.end),
        "mean_ccd_temp": _r(s.mean_temp), "folders": s.folders,
        "files": [f.rel_path for f in s.frames],
    }


def _match_dict(m: Match | None) -> dict[str, Any] | None:
    if m is None:
        return None
    return {"level": m.level, "sets": [s.id for s in m.sets], "n_frames": m.n_frames, "reasons": m.reasons}


def _group_dict(g: LightGroup) -> dict[str, Any]:
    return {
        "session": g.session.isoformat(), "label": g.label, "filter": g.filter,
        "sensor_key": g.sensor_key._asdict(), "exptime": g.exptime, "n": g.n,
        "start_utc": _iso(g.start), "end_utc": _iso(g.end), "mean_ccd_temp": _r(g.mean_temp),
        "level": g.level, "dark": _match_dict(g.dark), "flat": _match_dict(g.flat),
        "darkflat": _match_dict(g.darkflat), "files": [f.rel_path for f in g.frames],
    }


def _todo_dict(t: DarkTodo) -> dict[str, Any]:
    return {"kind": t.kind, "sensor_key": t.sensor_key._asdict(), "exptime": t.exptime,
            "set_temp": _r(t.set_temp, 1), "mean_ccd_temp": _r(t.mean_temp), "filters": t.filters,
            "sessions": t.sessions, "current_level": t.current_level, "reason": t.reason}


def inventory_to_dict(inv: Inventory) -> dict[str, Any]:
    levels = inv.levels_by_session_filter()
    return {
        "root": str(inv.root),
        "created": _iso(inv.created),
        "counts": dict(inv.counts_by_type()),
        "sensors": [{
            "sensor_key": s.key._asdict(), "label": s.key.label(), "n_frames": s.n_frames,
            "egain_header": s.egain, "adu_scale": s.adu_scale, "adu_scale_source": s.adu_scale_source,
            "effective_gain_e_per_adu": _r(s.effective_gain, 5), "saturation_adu": s.saturation_adu,
            "xpixsz_um": s.xpixsz, "focallen_mm": s.focallen,
            "pixel_scale_arcsec": _r(s.pixel_scale_arcsec, 3),
        } for s in inv.sensors],
        "levels": [{"session": k[0].isoformat(), "label": inv.session_label(k[0]), "filter": k[1],
                    "level": v} for k, v in sorted(levels.items())],
        "light_groups": [_group_dict(g) for g in inv.light_groups],
        "calibration_sets": {k: [_set_dict(s) for s in v] for k, v in inv.calibration_sets.items()},
        "darks_todo": [_todo_dict(t) for t in darks_todo(inv)],
        "anomalies": [{"severity": a.severity, "code": a.code, "message": a.message, "files": a.files}
                      for a in inv.anomalies],
        "ignored_dirs": inv.ignored_dirs,
        "empty_dirs": inv.empty_dirs,
        "other_extensions": inv.other_extensions,
    }
