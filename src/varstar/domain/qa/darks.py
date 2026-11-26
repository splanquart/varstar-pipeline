"""Dark levels per series and the darks actually combined for a light group (D16).

``stats`` maps ``str(frame.path)`` to the quick statistics of that frame
(``pixels.frame_quick_stats``), measured by the adapters.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import numpy as np

from ..inventory import nearest_night, set_night
from ..model import DARK, FrameSet, Inventory, LightGroup
from ..settings import MatchingSettings, Settings
from .model import offset_hours

FrameStats = Mapping[str, Mapping[str, Any]]


def set_summary(s: FrameSet, stats: FrameStats) -> dict[str, Any]:
    st = [stats[str(f.path)] for f in s.frames]
    return {
        "level": float(np.median([x["median"] for x in st])),
        "sigma": float(np.median([x["robust_sigma"] for x in st])),
        "p005": float(np.median([x["p005"] for x in st])),
        "corners": {k: float(np.median([x["corners"][k] for x in st])) for k in st[0]["corners"]},
        "block_range": float(np.median([x["block_range"] for x in st])),
    }


def level_clusters(levels: list[tuple[Any, float]], jump: float) -> list[list[Any]]:
    """Split items into groups whose levels are within `jump` of their neighbours."""
    ordered = sorted(levels, key=lambda kv: kv[1])
    clusters: list[list[tuple[Any, float]]] = []
    for item in ordered:
        if clusters and item[1] - clusters[-1][-1][1] <= jump:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return [[k for k, _ in c] for c in clusters]


def classify_excess(delta_adu: float, sigma_ref: float, sigma_new: float, gain_e_per_adu: float | None) -> str:
    """Offset change (noise unchanged) or extra light (shot noise grows as delta/gain)?"""
    if gain_e_per_adu is None or delta_adu <= 0:
        return "indéterminé"
    expected = delta_adu / gain_e_per_adu  # extra variance (ADU²) if the excess were photons
    measured = sigma_new ** 2 - sigma_ref ** 2
    ratio = measured / expected if expected > 0 else 0.0
    if ratio < 0.3:
        return "offset"
    if ratio > 0.7:
        return "lumière"
    return "indéterminé"


def dark_candidates(inv: Inventory, g: LightGroup, m: MatchingSettings) -> list[FrameSet]:
    """All dark series compatible with the darks chosen for `g` (same sensor key, exposure, temperature)."""
    if not g.dark or not g.dark.sets:
        return []
    exp = g.dark.sets[0].exptime
    ttol = m.temp_tolerance_c
    gt = g.mean_temp
    return [s for s in inv.calibration_sets.get(DARK, []) if s.sensor_key == g.sensor_key
            and abs(s.exptime - exp) < 1e-6
            and (gt is None or s.mean_temp is None or abs(s.mean_temp - gt) <= ttol)]


def usual_level_dark_sets(inv: Inventory, g: LightGroup, stats: FrameStats,
                          cfg: Settings) -> tuple[list[FrameSet], str | None]:
    """Without same-night darks, do not borrow the darks of a night whose offset is anomalous (P1).

    The darks of the nearest night at the usual level (the level shared by most dark frames) are
    used instead; the group stays at level B.
    """
    if not g.dark or not g.dark.sets:
        return [], None
    off = offset_hours(cfg)
    chosen = list(g.dark.sets)
    if any(set_night(s, off) == g.session for s in chosen):
        return chosen, None
    cands = dark_candidates(inv, g, cfg.matching)
    if len(cands) < 2:
        return chosen, None
    summ = {s.id: set_summary(s, stats) for s in cands}
    clusters = level_clusters([(s.id, summ[s.id]["level"]) for s in cands], cfg.qa.offset_jump_adu)
    if len(clusters) < 2:
        return chosen, None
    usual = max(clusters, key=lambda c: sum(s.n for s in cands if s.id in c))
    if all(s.id in usual for s in chosen):
        return chosen, None
    pool = [s for s in cands if s.id in usual]
    night, sets, gap = nearest_night(pool, g.start, g.end, off)
    old_night = set_night(chosen[0], off)
    note = (f"darks de la nuit du {old_night:%d/%m} écartés (offset anormal, contrôle P1 : "
            f"{np.median([summ[s.id]['level'] for s in chosen]):.1f} ADU) → darks de la nuit du {night:%d/%m} "
            f"au niveau habituel ({np.median([summ[s.id]['level'] for s in sets]):.1f} ADU, écart "
            f"{gap.total_seconds() / 86400:.1f} j)")
    return sets, note


def effective_dark_sets(inv: Inventory, g: LightGroup, stats: FrameStats,
                        cfg: Settings) -> tuple[list[FrameSet], list[str]]:
    """Darks actually combined for `g`: usual level when borrowed, single level within a night."""
    sets, note = usual_level_dark_sets(inv, g, stats, cfg)
    notes = [note] if note else []
    sets, note2 = consistent_dark_sets(sets, stats, cfg.qa.offset_jump_adu, g.start)
    if note2:
        notes.append(note2)
    return sets, notes


def consistent_dark_sets(sets: list[FrameSet], stats: FrameStats, jump: float,
                         t_ref: dt.datetime) -> tuple[list[FrameSet], str | None]:
    """Never combine darks of different levels: keep the level group nearest in time to the lights."""
    if len(sets) < 2:
        return sets, None
    summ = {s.id: set_summary(s, stats) for s in sets}
    clusters = level_clusters([(s.id, summ[s.id]["level"]) for s in sets], jump)
    if len(clusters) == 1:
        return sets, None
    nearest = min(sets, key=lambda s: abs((s.mid - t_ref).total_seconds()))
    keep_ids = next(c for c in clusters if nearest.id in c)
    kept = [s for s in sets if s.id in keep_ids]
    dropped = [s for s in sets if s.id not in keep_ids]
    note = ("séries de darks de niveaux différents : " + ", ".join(
        f"{s.id} {summ[s.id]['level']:.1f} ADU" for s in sets) + f" → seules {', '.join(s.id for s in kept)} "
        f"(les plus proches des lights) sont combinées ; écartées : {', '.join(s.id for s in dropped)}")
    return kept, note

