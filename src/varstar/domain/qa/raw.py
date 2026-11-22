"""Raw pixel checks (P*): from the quick statistics of the raw frames (measured by the adapters).

The flat comparison (P4) needs binned flats: ``flat_comparison_series`` says which series to
compare, the adapters build the flats, ``flat_pair_check`` judges each pair.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..inventory import match_darkflats, set_night
from ..model import DARK, DARKFLAT, FLAT, Frame, FrameSet, Inventory
from .darks import FrameStats, classify_excess, effective_dark_sets, level_clusters, set_summary
from .model import QAItem, group_label, night_label, offset_hours

DEFAULT_SATURATION_ADU = 65532.0


@dataclass
class RawChecks:
    items: list[QAItem]
    tables: dict[str, Any]
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)   # dark/darkflat set id → set_summary


def saturation_adu(inv: Inventory) -> float:
    sensor = inv.sensors[0] if inv.sensors else None
    return float(sensor.saturation_adu) if sensor and sensor.saturation_adu else DEFAULT_SATURATION_ADU


def _dark_sets(inv: Inventory) -> list[FrameSet]:
    return inv.calibration_sets.get(DARK, []) + inv.calibration_sets.get(DARKFLAT, [])


def sample_lights(inv: Inventory) -> list[Frame]:
    """First, middle and last light of every group."""
    out: list[Frame] = []
    for g in inv.light_groups:
        fr = g.frames
        out += sorted({fr[0].rel_path: fr[0], fr[len(fr) // 2].rel_path: fr[len(fr) // 2],
                       fr[-1].rel_path: fr[-1]}.values(), key=lambda f: f.date_obs)
    return out


def frames_to_measure(inv: Inventory) -> list[Frame]:
    """Frames whose quick statistics the raw checks need: every calibration frame + sample lights."""
    flat_sets = inv.calibration_sets.get(FLAT, [])
    return [f for s in _dark_sets(inv) + flat_sets for f in s.frames] + sample_lights(inv)


def raw_pixel_checks(inv: Inventory, cfg: dict[str, Any], stats: FrameStats) -> RawChecks:
    """P1, P2, P3, P7 and P8 (unsorted, P4 is added by the caller)."""
    q = cfg["qa"]
    off = offset_hours(cfg)
    sensor = inv.sensors[0] if inv.sensors else None
    saturation = saturation_adu(inv)
    gain = sensor.effective_gain if sensor else None
    items: list[QAItem] = []
    tables: dict[str, Any] = {}

    dark_sets = _dark_sets(inv)
    flat_sets = inv.calibration_sets.get(FLAT, [])
    sample = sample_lights(inv)
    all_frames = frames_to_measure(inv)
    summaries = {s.id: set_summary(s, stats) for s in dark_sets}

    # Table of dark levels per night and exposure (the 2021 case).
    rows = []
    for s in sorted(dark_sets, key=lambda s: (s.kind, s.exptime, s.start)):
        sm = summaries[s.id]
        rows.append({"set": s.id, "kind": s.kind, "exptime": s.exptime, "night": set_night(s, off).isoformat(),
                     "n": s.n, "level_adu": round(sm["level"], 1), "sigma_adu": round(sm["sigma"], 2),
                     "mean_ccd_temp": round(s.mean_temp, 2) if s.mean_temp is not None else None,
                     "folders": s.folders})
    tables["dark_levels"] = rows

    # P1 / P2 (set level): same exposure, different nights or series.
    jump = float(q["offset_jump_adu"])
    by_exp: dict[tuple, list[FrameSet]] = defaultdict(list)
    for s in dark_sets:
        by_exp[(s.sensor_key, s.exptime)].append(s)
    for (_, exp), sets in sorted(by_exp.items(), key=lambda kv: kv[0][1]):
        if len(sets) < 2:
            continue
        clusters = level_clusters([(s.id, summaries[s.id]["level"]) for s in sets], jump)
        if len(clusters) < 2:
            continue
        ref_ids = max(clusters, key=lambda c: sum(next(s.n for s in sets if s.id == i) for i in c))
        ref = [s for s in sets if s.id in ref_ids]
        ref_level = float(np.median([summaries[s.id]["level"] for s in ref]))
        ref_sigma = float(np.median([summaries[s.id]["sigma"] for s in ref]))
        for c in clusters:
            if c is ref_ids:
                continue
            members = [s for s in sets if s.id in c]
            lvl = float(np.median([summaries[s.id]["level"] for s in members]))
            sig = float(np.median([summaries[s.id]["sigma"] for s in members]))
            delta = lvl - ref_level
            kind = classify_excess(abs(delta), ref_sigma if delta > 0 else sig, sig if delta > 0 else ref_sigma, gain)
            nights = sorted({set_night(s, off) for s in members})
            nights_txt = ", ".join(night_label(inv, n) for n in nights)
            corner_diff = max(abs(summaries[s.id]["corners"][k] - summaries[ref[0].id]["corners"][k])
                              for s in members for k in summaries[s.id]["corners"])
            data = {"exptime": exp, "sets": c, "reference_sets": ref_ids, "delta_adu": round(delta, 2),
                    "sigma_ref_adu": round(ref_sigma, 2), "sigma_adu": round(sig, 2), "classification": kind,
                    "corner_diff_adu": round(corner_diff, 2)}
            files = [f.rel_path for s in members for f in s.frames]
            npix = min(stats[str(f.path)].get("corner_npix", 1) for s in members for f in s.frames)
            corner_err = ref_sigma / math.sqrt(max(npix, 1)) / math.sqrt(max(min(s.n for s in members), 1))
            nonuniform = delta > 0 and corner_diff > max(jump, 2.0, 5 * corner_err)
            if kind == "lumière" or nonuniform:
                items.append(QAItem(
                    "P2", "warning",
                    f"darks {exp:g} s, nuit(s) du {nights_txt} : {delta:+.1f} ADU par rapport aux autres nuits "
                    f"({ref_level:.1f} ADU), bruit {sig:.2f} contre {ref_sigma:.2f} ADU, excès non uniforme "
                    f"jusqu'à {corner_diff:.1f} ADU dans les coins → fuite de lumière probable", files,
                    "dark trop haut → fond calibré trop bas, flux des étoiles faibles biaisés",
                    "couvercle opaque, caméra recouverte, darks dans le noir complet", data))
            else:
                verdict = {"offset": "saut d'offset électronique (bruit inchangé, écart uniforme)",
                           "indéterminé": "cause indéterminée"}.get(kind, kind)
                items.append(QAItem(
                    "P1", "warning",
                    f"darks {exp:g} s, nuit(s) du {nights_txt} : niveau {lvl:.1f} ADU contre {ref_level:.1f} ADU "
                    f"les autres nuits ({delta:+.1f} ADU, bruit {sig:.2f} contre {ref_sigma:.2f} ADU) → {verdict}. "
                    "Ces darks ne sont jamais combinés avec ceux des autres nuits", files,
                    "si on les combinait avec ceux des autres nuits, le master serait faux pour toutes les nuits ; "
                    "en photométrie d'ouverture un décalage uniforme est surtout retiré par le fond local (anneau), "
                    "mais il fausse la mise à l'échelle des darks (repli B), les contrôles de fond et la traçabilité",
                    "fixer l'offset dans le profil N.I.N.A., vérifier qu'il est écrit dans les en-têtes (OFFSET), "
                    "et faire des darks la même nuit que les lights", data))

    # P2 (frame level): single darks that deviate from their series.
    fx = float(q["leak_frame_excess_adu"])
    for s in dark_sets:
        sm = summaries[s.id]
        lv = np.array([stats[str(f.path)]["median"] for f in s.frames])
        lim = max(fx, 5 * 1.4826 * float(np.median(np.abs(lv - np.median(lv))))) if s.n >= 5 else fx
        odd = [(f, x - sm["level"]) for f, x in zip(s.frames, lv) if abs(x - sm["level"]) > lim]
        if odd and len(odd) < s.n:
            dmax = max(odd, key=lambda x: abs(x[1]))[1]
            items.append(QAItem("P2", "warning", f"{s.id} : {len(odd)}/{s.n} pose(s) s'écartent du niveau de la série "
                                f"(jusqu'à {dmax:+.1f} ADU) — lumière parasite ou dérive de température",
                                [f.rel_path for f, _ in odd],
                                "le rejet du master les atténue mais la série n'est pas homogène",
                                "darks dans le noir complet, température stabilisée",
                                {"max_delta_adu": round(dmax, 2)}))

    # P3: flat exposure (per set).
    lo_f, hi_f = float(cfg["masters"]["flat_level_min_frac"]), float(cfg["masters"]["flat_level_max_frac"])
    flat_rows = []
    for s in flat_sets:
        st = [stats[str(f.path)] for f in s.frames]
        fr = [x["median"] / saturation for x in st]
        sat = max(x["frac_saturated"] for x in st)
        flat_rows.append({"set": s.id, "filter": s.filter, "n": s.n, "level_min_frac": round(min(fr), 3),
                          "level_max_frac": round(max(fr), 3), "max_frac_saturated": sat})
        bad = [f.rel_path for f, x in zip(s.frames, fr) if not lo_f <= x <= hi_f]
        if bad or sat > 1e-3:
            items.append(QAItem("P3", "warning",
                                f"{s.id} : niveau {100 * min(fr):.0f}–{100 * max(fr):.0f} % de la dynamique"
                                + (f", jusqu'à {100 * sat:.2f} % de pixels saturés" if sat > 1e-3 else ""),
                                bad, "flat bruité (sous-exposé) ou non linéaire (surexposé)",
                                f"viser {100 * lo_f:.0f}–{100 * hi_f:.0f} % (≈ 50 %) avec l'assistant de flats de N.I.N.A."))
    tables["flat_levels"] = flat_rows

    # P7: ADU scale on every frame read.
    if sensor and sensor.adu_scale:
        odd = [f.rel_path for f in all_frames if stats[str(f.path)]["adu_scale"] != sensor.adu_scale]
        if odd:
            items.append(QAItem("P7", "warning", f"{len(odd)} image(s) avec un facteur ADU différent de "
                                f"×{sensor.adu_scale}", odd, "gain effectif faux pour ces images",
                                "ne pas changer le mode de lecture (bits) de la caméra en cours de projet"))

    # P8: lights vs their darks (raw sky estimate = light median − dark level).
    thr = float(q["dark_above_light_adu"])
    sky_rows = []
    for g in inv.light_groups:
        if not g.dark or not g.dark.sets:
            continue
        if any(abs(s.exptime - g.exptime) > cfg["matching"]["exptime_tolerance_s"] for s in g.dark.sets):
            continue
        ls = [stats[str(f.path)]["median"] for f in sample if f in g.frames]
        if not ls:
            continue
        eff, _ = effective_dark_sets(inv, g, stats, cfg)
        dl = float(np.median([summaries[s.id]["level"] for s in eff]))
        sky = float(np.median(ls)) - dl
        sky_rows.append({"group": group_label(g), "light_median_adu": round(float(np.median(ls)), 1),
                         "dark_level_adu": round(dl, 1), "raw_sky_adu": round(sky, 1), "dark_level": g.dark.level})
        if sky < thr:
            items.append(QAItem(
                "P8", "error" if sky < 0 else "warning",
                f"{group_label(g)} : médiane des lights {np.median(ls):.1f} ADU, darks {dl:.1f} ADU → fond de ciel "
                f"estimé {sky:+.1f} ADU : les darks sont trop hauts (offset différent des lights ?)",
                [f.rel_path for f in g.frames[:1]],
                "fond calibré négatif, flux faussés (surtout étoiles faibles)",
                "faire des darks la même nuit que les lights, offset fixé", {"raw_sky_adu": round(sky, 2)}))
    tables["raw_sky"] = sky_rows
    return RawChecks(items, tables, summaries)


# --------------------------------------------------------------------------- P4: successive flats


def flat_comparison_series(inv: Inventory) -> list[tuple[str | None, list[FrameSet]]]:
    """Flat series of the same sensor and filter, in time order (only filters with several series)."""
    by_filter: dict[tuple, list[FrameSet]] = defaultdict(list)
    for s in inv.calibration_sets.get(FLAT, []):
        by_filter[(s.sensor_key, s.filter)].append(s)
    out = []
    for (_, filt), sets in sorted(by_filter.items(), key=lambda kv: kv[0][1] or ""):
        sets = sorted(sets, key=lambda s: s.start)
        if len(sets) >= 2:
            out.append((filt, sets))
    return out


def flat_dark_level(s: FrameSet, inv: Inventory, cfg: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> float:
    """Level of the darks of flat matched to a flat series (0 when there is none)."""
    m = match_darkflats(s, inv.calibration_sets.get(DARKFLAT, []), inv.calibration_sets.get(DARK, []), cfg)
    return float(np.median([summaries[x.id]["level"] for x in m.sets])) if m.sets else 0.0


def flat_pair_check(inv: Inventory, cfg: dict[str, Any], filt: str | None, a: FrameSet, b: FrameSet,
                    met: dict[str, float], ratio_file: str) -> tuple[QAItem | None, dict[str, Any]]:
    """Table row and finding (if any) for two successive flats; ``met`` from ``flat_ratio_metrics``."""
    q = cfg["qa"]
    moved_dust = met["dust_pct"] > float(q["flat_dust_pct"])
    moved_shape = met["shape_pct"] > float(q["flat_shape_pct"])
    verdict = ("poussières déplacées" if moved_dust else "") + (
        (" et " if moved_dust else "") + "forme différente (vignetage)" if moved_shape else "")
    row = {"filter": filt, "a": a.id, "b": b.id, "a_labels": a.folder_labels, "b_labels": b.folder_labels,
           **met, "verdict": verdict or "identiques"}
    if not verdict:
        return None, row
    la = ", ".join(a.folder_labels) or f"{a.start:%d/%m}"
    lb = ", ".join(b.folder_labels) or f"{b.start:%d/%m}"
    shared = [g for g in inv.light_groups if g.filter == filt and g.flat and g.flat.sets
              and (g.flat.reasons or g.flat.level == "B")
              and g.flat.sets[0].id in (a.id, b.id)]
    sev = "warning" if moved_dust else "info"
    return QAItem(
        "P4", sev,
        f"flats {filt} du {a.start:%d/%m} ({la}) et du {b.start:%d/%m} ({lb}) : {verdict} "
        f"(structure locale {met['dust_pct']:.2f} %, forme {met['shape_pct']:.2f} %, bruit {met['noise_pct']:.2f} %)"
        + (" — utilisés pour " + ", ".join(group_label(g) for g in shared) if shared else ""),
        [ratio_file],
        ("poussières déplacées : un flat d'une autre session laisse des anneaux → flux faux pour les "
         "étoiles concernées" if moved_dust else
         "forme du flat différente (souvent l'éclairage du panneau ou du ciel) : erreur lente avec la "
         "position dans le champ, en partie absorbée par la photométrie différentielle (terme de champ)"),
        "faire des flats à chaque session sans démonter le train optique entre lights et flats ; "
        "nettoyer le filtre/capteur seulement juste avant une nouvelle série de flats", row), row
