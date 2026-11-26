"""Early quality checks (docs/QA_CHECKS.md): detect acquisition errors before any flux is measured.

Three stages, cheapest first:
- header checks (H*), from the inventory only;
- raw pixel checks (P*), quick sub-sampled statistics of the raw frames (cached per file);
- post-calibration checks (C*), on the calibrated lights.

Nothing is ever corrected here: every finding is reported with numbers, the files concerned,
its likely photometric impact and an advice for the next observing session.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits

from .domain.units import detect_adu_scale
from .domain.model import DARK, DARKFLAT, FLAT, LIGHT, Frame, FrameSet, Inventory, LightGroup
from .domain.inventory import match_darkflats, nearest_night, set_night

log = logging.getLogger("varstar")

SEVERITIES = ("error", "warning", "info")
SEVERITY_FR = {"error": "erreur", "warning": "avertissement", "info": "info"}
STATS_VERSION = 3

TITLES = {
    "H1": "Dossiers incohérents avec les en-têtes",
    "H2": "OFFSET absent des en-têtes",
    "H3": "Température du capteur",
    "H4": "Version du logiciel d'acquisition",
    "H5": "Mise au point des flats",
    "H6": "Pointage",
    "H7": "Météo",
    "H8": "Darks pris télescope pointé vers le ciel",
    "H9": "Nombre de poses de calibration",
    "P1": "Niveau (offset) des darks",
    "P2": "Fuite de lumière dans les darks",
    "P3": "Exposition des flats",
    "P4": "Comparaison des flats successifs",
    "P7": "Facteur ADU",
    "P8": "Niveau des lights comparé aux darks",
    "C1": "Résidu d'amp glow après calibration",
    "C3": "Fond calibré négatif ou pixels invalides",
}


@dataclass
class QAItem:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    files: list[str] = field(default_factory=list)
    impact: str = ""
    advice: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sorted(items: Iterable[QAItem]) -> list[QAItem]:
    return sorted(items, key=lambda i: (SEVERITIES.index(i.severity), i.check))


def _night_label(inv: Inventory, night: dt.date) -> str:
    labels = {g.label for g in inv.light_groups if g.session == night and g.label}
    return f"{night:%d/%m}" + (f" ({', '.join(sorted(labels))})" if labels else "")


def _group_label(g: LightGroup) -> str:
    return f"{g.label or g.session.isoformat()} / {g.filter}"


def _offset_hours(cfg: dict[str, Any]) -> float:
    return float(cfg["inventory"].get("session_day_offset_hours", -12))


# =========================================================================== header checks


def header_checks(inv: Inventory, cfg: dict[str, Any]) -> list[QAItem]:
    q = cfg["qa"]
    items: list[QAItem] = []
    off = _offset_hours(cfg)

    # H1 / H2: mirrored from the inventory anomalies.
    folder = [a for a in inv.anomalies if a.code.startswith("folder_")]
    if folder:
        items.append(QAItem("H1", "warning",
                            f"{sum(len(a.files) for a in folder)} fichier(s) rangé(s) dans un dossier qui contredit "
                            "l'en-tête (l'en-tête fait foi)", [f for a in folder for f in a.files],
                            "aucun (le pipeline classe par en-tête)",
                            "ranger les fichiers selon le type, la pose et le filtre réels",
                            {"details": [a.message for a in folder]}))
    if any(a.code == "offset_missing" for a in inv.anomalies):
        items.append(QAItem("H2", "warning", "OFFSET absent de tous les en-têtes : impossible de vérifier que darks "
                            "et lights ont le même offset (seul le contrôle P1 sur les pixels peut le voir)", [],
                            "un dark d'offset différent décale tout le fond calibré (retiré en grande partie par le "
                            "fond local, mais fausse la mise à l'échelle des darks)",
                            "utiliser un pilote qui écrit OFFSET (pilote natif ZWO dans N.I.N.A. plutôt qu'ASCOM) "
                            "et fixer l'offset dans le profil de la caméra"))

    # H3: cooling.
    tol = float(q["ccd_temp_delta_c"])
    by_series: dict[tuple[str, dt.date], list[Frame]] = defaultdict(list)
    for f in inv.frames:
        if f.date_obs is not None:
            by_series[(f.imagetyp, f.session_date(off))].append(f)
    for (kind, night), frames in sorted(by_series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        frames = sorted(frames, key=lambda f: f.date_obs)
        bad = [(i, f) for i, f in enumerate(frames) if f.ccd_temp is not None and f.set_temp is not None
               and abs(f.ccd_temp - f.set_temp) > tol]
        if not bad:
            continue
        dmax = max(abs(f.ccd_temp - f.set_temp) for _, f in bad)
        first = sum(1 for k, (i, _) in enumerate(bad) if i == k)
        if first == len(frames):
            where = " — toute la nuit (consigne non tenue)"
        elif first:
            where = f", dont les {first} première(s) de la nuit (refroidissement pas encore stabilisé)"
        else:
            where = ""
        kind_fr = {LIGHT: "lights", DARK: "darks", FLAT: "flats", DARKFLAT: "darks de flat"}.get(kind, kind)
        items.append(QAItem(
            "H3", "warning" if kind in (LIGHT, DARK) else "info",
            f"{kind_fr}, nuit du {_night_label(inv, night)} : {len(bad)}/{len(frames)} pose(s) à plus de "
            f"{tol:g} °C de la consigne (écart max {dmax:.1f} °C){where}",
            [f.rel_path for _, f in bad],
            "courant d'obscurité et pixels chauds différents de ceux du dark → résidus dans le fond",
            "attendre la stabilisation de la température avant de lancer la séquence (attente après "
            "« Cool Camera » dans N.I.N.A.) ; en été, choisir une consigne que la caméra tient",
            {"n_bad": len(bad), "n": len(frames), "max_delta_c": round(dmax, 2)}))

    # H4: acquisition software per night, and mismatch between lights and their calibration.
    versions: dict[dt.date, set[str]] = defaultdict(set)
    for f in inv.frames:
        if f.swcreate and f.date_obs is not None:
            versions[f.session_date(off)].add(f.swcreate)
    if len({v for vs in versions.values() for v in vs}) > 1:
        items.append(QAItem("H4", "info", "le logiciel d'acquisition change selon les nuits : "
                            + " ; ".join(f"{n:%d/%m} {', '.join(sorted(v))}" for n, v in sorted(versions.items())),
                            [], "une mise à jour peut réinitialiser les réglages de la caméra (offset, gain)",
                            "après une mise à jour de N.I.N.A. ou d'un pilote, vérifier gain/offset et refaire "
                            "des darks", {"by_night": {n.isoformat(): sorted(v) for n, v in versions.items()}}))
    mismatched = []
    for g in inv.light_groups:
        lv = {f.swcreate for f in g.frames if f.swcreate}
        cv = {f.swcreate for s in (g.dark.sets if g.dark else []) for f in s.frames if f.swcreate}
        if lv and cv and not (cv <= lv):
            mismatched.append(f"{_group_label(g)} (lights {', '.join(sorted(lv))}, darks {', '.join(sorted(cv))})")
    if mismatched:
        items.append(QAItem("H4", "info", f"{len(mismatched)} groupe(s) dont les darks viennent d'une autre version "
                            "du logiciel : " + " ; ".join(mismatched), [],
                            "réglages de caméra éventuellement différents (offset)", ""))

    # H5: focus position of the flats vs the lights.
    ftol = float(q["focpos_tolerance_steps"])
    for g in inv.light_groups:
        if not g.flat or not g.flat.sets:
            continue
        lp = [f.focpos for f in g.frames if f.focpos is not None]
        fp = [f.focpos for s in g.flat.sets for f in s.frames if f.focpos is not None]
        if not lp or not fp:
            continue
        fmed = statistics.median(fp)
        lo, hi = min(lp), max(lp)
        if lo - ftol <= fmed <= hi + ftol:
            continue
        dist = min(abs(fmed - lo), abs(fmed - hi))
        other = g.flat.level != "A" or bool(g.flat.reasons)
        items.append(QAItem(
            "H5", "warning" if other else "info",
            f"{_group_label(g)} : flats pris à FOCPOS {fmed:.0f}, lights entre {lo:.0f} et {hi:.0f} "
            f"(écart {dist:.0f} pas)" + (" — flats d'une autre session" if other else ""),
            [f.rel_path for f in g.flat.sets[0].frames[:1]],
            "si le tube de mise au point bouge beaucoup, les ombres de poussières se déplacent et le flat "
            "les corrige mal (vérifié par P4)",
            "faire les flats sans toucher à la mise au point, ou à la position de mise au point de la nuit",
            {"flat_focpos": fmed, "light_focpos": [lo, hi]}))

    # H6: pointing.
    sensor = inv.sensors[0] if inv.sensors else None
    scale = sensor.pixel_scale_arcsec if sensor else None
    by_obj: dict[str, list[LightGroup]] = defaultdict(list)
    for g in inv.light_groups:
        by_obj[g.frames[0].object or "?"].append(g)
    if scale:
        for obj, groups in by_obj.items():
            pos = [(f.ra, f.dec) for g in groups for f in g.frames if f.ra is not None and f.dec is not None]
            if len(pos) < 2:
                continue
            ra0 = statistics.median(p[0] for p in pos)
            de0 = statistics.median(p[1] for p in pos)
            sk = groups[0].sensor_key
            field_arcmin = min(sk.naxis1, sk.naxis2) * scale / 60
            limit = float(q["pointing_max_field_frac"]) * field_arcmin
            for g in groups:
                far = []
                for f in g.frames:
                    if f.ra is None or f.dec is None:
                        continue
                    d = 60 * math.hypot((f.ra - ra0) * math.cos(math.radians(de0)), f.dec - de0)
                    if d > limit:
                        far.append((f, d))
                if far:
                    dmax = max(d for _, d in far)
                    items.append(QAItem(
                        "H6", "warning",
                        f"{_group_label(g)} ({obj}) : {len(far)}/{g.n} pose(s) pointée(s) à plus de {limit:.0f}′ du "
                        f"centre habituel (max {dmax:.0f}′, champ {field_arcmin:.0f}′)",
                        [f.rel_path for f, _ in far],
                        "étoiles de comparaison différentes d'une session à l'autre, bords du champ (vignetage)",
                        "recentrer par résolution astrométrique (« Center » dans N.I.N.A.) au début de chaque "
                        "cible et après un retournement", {"max_arcmin": round(dmax, 1)}))

    # H7: weather (lights only).
    cmax, margin = float(q["cloud_cover_max"]), float(q["dew_margin_c"])
    sessions: dict[dt.date, list[Frame]] = defaultdict(list)
    for g in inv.light_groups:
        sessions[g.session].extend(g.frames)
    for night, frames in sorted(sessions.items()):
        cloudy = [f for f in frames if f.cloudcvr is not None and f.cloudcvr > cmax]
        dew = [f for f in frames if f.ambtemp is not None and f.dewpoint is not None
               and f.ambtemp - f.dewpoint < margin]
        if cloudy:
            items.append(QAItem("H7", "info", f"nuit du {_night_label(inv, night)} : couverture nuageuse annoncée "
                                f"> {cmax:g} % sur {len(cloudy)}/{len(frames)} pose(s) (source météo souvent en ligne : "
                                "indicatif)", [f.rel_path for f in cloudy],
                                "extinction variable : à confirmer par le point zéro (contrôle qualité M3)", ""))
        if dew:
            mn = min(f.ambtemp - f.dewpoint for f in dew)
            items.append(QAItem("H7", "warning", f"nuit du {_night_label(inv, night)} : température à moins de "
                                f"{margin:g} °C du point de rosée sur {len(dew)}/{len(frames)} pose(s) (min {mn:.1f} °C)",
                                [f.rel_path for f in dew],
                                "buée possible sur l'optique : baisse de flux et halos, lents et non corrigés par le flat",
                                "activer la résistance chauffante et le pare-buée dès le début de la nuit"))

    # H8: darks taken with the telescope pointing at the sky.
    alt_darks = [f for f in inv.frames if f.imagetyp in (DARK, DARKFLAT) and f.centalt is not None
                 and f.centalt > 10]
    if alt_darks:
        alts = [f.centalt for f in alt_darks]
        items.append(QAItem("H8", "info", f"{len(alt_darks)} dark(s) pris télescope pointé vers le ciel "
                            f"(altitude {min(alts):.0f}–{max(alts):.0f}°) : une fuite de lumière est possible, "
                            "voir P2", [], "fuite de lumière → dark trop haut, fond calibré trop bas",
                            "faire les darks couvercle opaque et caméra recouverte, ou caméra démontée"))

    # H9: number of calibration frames actually combined.
    for g in inv.light_groups:
        if g.dark and g.dark.sets and g.dark.n_frames < int(q["min_darks"]):
            items.append(QAItem("H9", "info", f"{_group_label(g)} : {g.dark.n_frames} darks seulement "
                                f"(< {q['min_darks']}) pour le master", [],
                                "master dark plus bruité, rejet des cosmiques moins robuste",
                                f"faire au moins {q['min_darks']} darks par pose et par nuit"))
        if g.flat and g.flat.sets and g.flat.n_frames < int(q["min_flats"]):
            items.append(QAItem("H9", "info", f"{_group_label(g)} : {g.flat.n_frames} flats seulement "
                                f"(< {q['min_flats']})", [], "flat plus bruité",
                                f"faire au moins {q['min_flats']} flats par filtre"))
    return _sorted(items)


# =========================================================================== raw pixel statistics


def _read_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw integer data as int64-able array, float32 ADU data) without any BZERO memmap issue."""
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
        hdu = hdul[0]
        raw = np.asarray(hdu.data)
        bzero = hdu.header.get("BZERO", 0)
        bscale = hdu.header.get("BSCALE", 1)
    if bscale == 1 and float(bzero) == int(bzero) and np.issubdtype(raw.dtype, np.integer):
        ints = raw.astype(np.int64) + int(bzero)
        return ints, ints.astype(np.float32)
    data = (raw * bscale + bzero).astype(np.float32)
    return data, data


def _rsig(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _clipped_mean(x: np.ndarray, med: float, sigma: float, k: float = 5.0) -> float:
    """Mean of the values within med ± k·sigma.

    Unlike the median, it is not quantised (the raw values are multiples of 4 ADU) and it ignores
    hot pixels and cosmic rays.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return float(np.mean(x))
    sel = x[np.abs(x - med) <= k * sigma]
    return float(np.mean(sel)) if sel.size else med


def quick_stats(path: str, step: int = 4, saturation: float = 65532.0) -> dict[str, Any]:
    """Sub-sampled statistics of one raw frame (central region avoids the amp glow in the corners)."""
    ints, data = _read_raw(path)
    sub = data[::step, ::step]
    h, w = sub.shape
    cy, cx = max(h // 8, 1), max(w // 8, 1)
    center = sub[cy:h - cy, cx:w - cx]
    med = float(np.median(center))
    sig = _rsig(center)
    level = _clipped_mean(center, med, sig)
    sel = center[np.abs(center - med) <= 5 * sig] if sig > 0 else center
    noise = float(np.std(sel))  # not quantised, unlike the MAD of multiples of 4 ADU
    corners = {
        "haut_gauche": sub[:cy, :cx], "haut_droit": sub[:cy, w - cx:],
        "bas_gauche": sub[h - cy:, :cx], "bas_droit": sub[h - cy:, w - cx:],
    }
    # 3x3 blocks of the central region: a light leak is rarely uniform.
    bh, bw = center.shape[0] // 3, center.shape[1] // 3
    blocks = [_clipped_mean(b, float(np.median(b)), sig) for b in
              (center[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] for i in range(3) for j in range(3))]
    return {
        "median": level,
        "robust_sigma": noise,
        "p005": float(np.percentile(center, 0.5)),
        "corners": {k: _clipped_mean(v, float(np.median(v)), sig) - level for k, v in corners.items()},
        "corner_npix": int(cy * cx),
        "block_range": max(blocks) - min(blocks),
        "frac_saturated": float(np.mean(sub >= saturation)),
        "adu_scale": detect_adu_scale(ints[::step, ::step]),
        "shape": list(data.shape),
    }


def binned_frame(path: str, bin_px: int = 8) -> np.ndarray:
    """Full-resolution frame averaged in bin_px x bin_px blocks (for flat comparisons)."""
    _, data = _read_raw(path)
    h, w = (data.shape[0] // bin_px) * bin_px, (data.shape[1] // bin_px) * bin_px
    return data[:h, :w].reshape(h // bin_px, bin_px, w // bin_px, bin_px).mean(axis=(1, 3)).astype(np.float32)


def _file_id(f: Frame) -> str:
    st = f.path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


class StatsCache:
    """Per-file quick statistics, cached in <workdir>/qa/frame_stats.json (keyed by size+mtime)."""

    def __init__(self, workdir: Path, cfg: dict[str, Any], saturation: float):
        self.path = workdir / "qa" / "frame_stats.json"
        self.step = int(cfg["qa"].get("pixel_step", 4))
        self.saturation = saturation
        self.processes = max(1, int(cfg.get("runtime", {}).get("processes", 1)))
        self.data: dict[str, Any] = {}
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if raw.get("version") == STATS_VERSION and raw.get("step") == self.step:
                    self.data = raw["files"]
            except (json.JSONDecodeError, KeyError):
                self.data = {}

    def get(self, frames: list[Frame], progress: str | None = None) -> dict[str, dict[str, Any]]:
        todo = [f for f in frames if self.data.get(str(f.path), {}).get("id") != _file_id(f)]
        if todo:
            if progress:
                print(f"  {progress} : statistiques rapides de {len(todo)} image(s) brute(s) …", flush=True)
            paths = [str(f.path) for f in todo]
            if self.processes > 1 and len(todo) > 4:
                with ProcessPoolExecutor(self.processes) as ex:
                    results = list(ex.map(quick_stats, paths, [self.step] * len(paths),
                                          [self.saturation] * len(paths), chunksize=4))
            else:
                results = [quick_stats(p, self.step, self.saturation) for p in paths]
            for f, r in zip(todo, results):
                self.data[str(f.path)] = {"id": _file_id(f), **r}
            self.save()
        return {str(f.path): self.data[str(f.path)] for f in frames}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": STATS_VERSION, "step": self.step, "files": self.data}),
                       encoding="utf-8")
        tmp.replace(self.path)


def set_summary(s: FrameSet, stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
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


def dark_candidates(inv: Inventory, g: LightGroup, cfg: dict[str, Any]) -> list[FrameSet]:
    """All dark series compatible with the darks chosen for `g` (same sensor key, exposure, temperature)."""
    if not g.dark or not g.dark.sets:
        return []
    exp = g.dark.sets[0].exptime
    ttol = float(cfg["matching"]["temp_tolerance_c"])
    gt = g.mean_temp
    return [s for s in inv.calibration_sets.get(DARK, []) if s.sensor_key == g.sensor_key
            and abs(s.exptime - exp) < 1e-6
            and (gt is None or s.mean_temp is None or abs(s.mean_temp - gt) <= ttol)]


def usual_level_dark_sets(inv: Inventory, g: LightGroup, stats: dict[str, dict[str, Any]],
                          cfg: dict[str, Any]) -> tuple[list[FrameSet], str | None]:
    """Without same-night darks, do not borrow the darks of a night whose offset is anomalous (P1).

    The darks of the nearest night at the usual level (the level shared by most dark frames) are
    used instead; the group stays at level B.
    """
    if not g.dark or not g.dark.sets:
        return [], None
    off = _offset_hours(cfg)
    chosen = list(g.dark.sets)
    if any(set_night(s, off) == g.session for s in chosen):
        return chosen, None
    cands = dark_candidates(inv, g, cfg)
    if len(cands) < 2:
        return chosen, None
    summ = {s.id: set_summary(s, stats) for s in cands}
    clusters = level_clusters([(s.id, summ[s.id]["level"]) for s in cands], float(cfg["qa"]["offset_jump_adu"]))
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


def effective_dark_sets(inv: Inventory, g: LightGroup, stats: dict[str, dict[str, Any]],
                        cfg: dict[str, Any]) -> tuple[list[FrameSet], list[str]]:
    """Darks actually combined for `g`: usual level when borrowed, single level within a night."""
    sets, note = usual_level_dark_sets(inv, g, stats, cfg)
    notes = [note] if note else []
    sets, note2 = consistent_dark_sets(sets, stats, float(cfg["qa"]["offset_jump_adu"]), g.start)
    if note2:
        notes.append(note2)
    return sets, notes


def consistent_dark_sets(sets: list[FrameSet], stats: dict[str, dict[str, Any]], jump: float,
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


# =========================================================================== raw pixel checks


def raw_checks(inv: Inventory, cfg: dict[str, Any], workdir: Path) -> tuple[list[QAItem], dict[str, Any]]:
    q = cfg["qa"]
    off = _offset_hours(cfg)
    sensor = inv.sensors[0] if inv.sensors else None
    saturation = float(sensor.saturation_adu) if sensor and sensor.saturation_adu else 65532.0
    gain = sensor.effective_gain if sensor else None
    cache = StatsCache(workdir, cfg, saturation)
    items: list[QAItem] = []
    tables: dict[str, Any] = {}

    dark_sets = inv.calibration_sets.get(DARK, []) + inv.calibration_sets.get(DARKFLAT, [])
    flat_sets = inv.calibration_sets.get(FLAT, [])
    sample_lights = []
    for g in inv.light_groups:
        fr = g.frames
        sample_lights += sorted({fr[0].rel_path: fr[0], fr[len(fr) // 2].rel_path: fr[len(fr) // 2],
                                 fr[-1].rel_path: fr[-1]}.values(), key=lambda f: f.date_obs)
    all_frames = [f for s in dark_sets + flat_sets for f in s.frames] + sample_lights
    stats = cache.get(all_frames, progress="contrôles")
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
            nights_txt = ", ".join(_night_label(inv, n) for n in nights)
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
        ls = [stats[str(f.path)]["median"] for f in sample_lights if f in g.frames]
        if not ls:
            continue
        eff, _ = effective_dark_sets(inv, g, stats, cfg)
        dl = float(np.median([summaries[s.id]["level"] for s in eff]))
        sky = float(np.median(ls)) - dl
        sky_rows.append({"group": _group_label(g), "light_median_adu": round(float(np.median(ls)), 1),
                         "dark_level_adu": round(dl, 1), "raw_sky_adu": round(sky, 1), "dark_level": g.dark.level})
        if sky < thr:
            items.append(QAItem(
                "P8", "error" if sky < 0 else "warning",
                f"{_group_label(g)} : médiane des lights {np.median(ls):.1f} ADU, darks {dl:.1f} ADU → fond de ciel "
                f"estimé {sky:+.1f} ADU : les darks sont trop hauts (offset différent des lights ?)",
                [f.rel_path for f in g.frames[:1]],
                "fond calibré négatif, flux faussés (surtout étoiles faibles)",
                "faire des darks la même nuit que les lights, offset fixé", {"raw_sky_adu": round(sky, 2)}))
    tables["raw_sky"] = sky_rows

    # P4: successive flats of the same filter.
    if q.get("flat_compare", True) and flat_sets:
        p4_items, p4_rows = _flat_comparisons(inv, cfg, workdir, flat_sets, summaries)
        items += p4_items
        tables["flat_pairs"] = p4_rows
    return _sorted(items), tables


def _quick_flat(s: FrameSet, cfg: dict[str, Any], workdir: Path, dark_level: float) -> np.ndarray:
    q = cfg["qa"]
    nmax, b = int(q["flat_compare_max_frames"]), int(q["flat_compare_bin"])
    frames = sorted(s.frames, key=lambda f: f.date_obs)
    if len(frames) > nmax:
        idx = np.linspace(0, len(frames) - 1, nmax).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx))]
    key = f"{s.id}_{b}_{len(frames)}_{dark_level:.1f}_{_file_id(frames[0])}"
    cache = workdir / "qa" / "flats" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    procs = max(1, int(cfg.get("runtime", {}).get("processes", 1)))
    paths = [str(f.path) for f in frames]
    if procs > 1 and len(paths) > 2:
        with ProcessPoolExecutor(min(procs, len(paths))) as ex:
            stack = list(ex.map(binned_frame, paths, [b] * len(paths)))
    else:
        stack = [binned_frame(p, b) for p in paths]
    flat = np.median(np.stack(stack), axis=0) - np.float32(dark_level)
    flat = (flat / np.median(flat)).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, flat)
    return flat


def _block_median(img: np.ndarray, block: int) -> np.ndarray:
    h, w = (img.shape[0] // block) * block, (img.shape[1] // block) * block
    v = img[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(
        h // block, w // block, -1)
    return np.median(v, axis=2)


def flat_ratio_metrics(a: np.ndarray, b: np.ndarray, block: int = 16) -> dict[str, float]:
    """Compare two normalised (binned) flats.

    shape_pct: peak-to-peak of the large-scale ratio (vignetting moved: rotation, refocus, panel);
    dust_pct: strongest local structure once the large scale is removed (dust shadows moved).
    """
    r = a / b
    r = r / np.median(r)
    coarse = _block_median(r, block)
    shape = 100 * float(np.percentile(coarse, 99) - np.percentile(coarse, 1))
    up = np.kron(coarse, np.ones((block, block), dtype=np.float32))
    h, w = up.shape
    resid = r[:h, :w] / up - 1.0
    # 3x3 smoothing suppresses the pixel noise, dust shadows are several bins wide.
    small = _block_median(resid, 3)
    noise = 100 * _rsig(resid)
    dust = 100 * float(np.max(np.abs(small)))
    return {"shape_pct": round(shape, 3), "dust_pct": round(dust, 3), "noise_pct": round(noise, 3)}


def _flat_comparisons(inv: Inventory, cfg: dict[str, Any], workdir: Path, flat_sets: list[FrameSet],
                      summaries: dict[str, dict[str, Any]]) -> tuple[list[QAItem], list[dict[str, Any]]]:
    q = cfg["qa"]
    items: list[QAItem] = []
    rows: list[dict[str, Any]] = []
    darkflat_sets = inv.calibration_sets.get(DARKFLAT, [])
    darks = inv.calibration_sets.get(DARK, [])
    by_filter: dict[tuple, list[FrameSet]] = defaultdict(list)
    for s in flat_sets:
        by_filter[(s.sensor_key, s.filter)].append(s)
    used = {s.id: g for g in inv.light_groups if g.flat for s in g.flat.sets}
    for (_, filt), sets in sorted(by_filter.items(), key=lambda kv: kv[0][1] or ""):
        sets = sorted(sets, key=lambda s: s.start)
        if len(sets) < 2:
            continue
        print(f"  contrôles : comparaison des flats {filt} ({len(sets)} séries) …", flush=True)
        flats = {}
        for s in sets:
            m = match_darkflats(s, darkflat_sets, darks, cfg)
            level = float(np.median([summaries[x.id]["level"] for x in m.sets])) if m.sets else 0.0
            flats[s.id] = _quick_flat(s, cfg, workdir, level)
        for a, b in zip(sets, sets[1:]):
            met = flat_ratio_metrics(flats[a.id], flats[b.id])
            moved_dust = met["dust_pct"] > float(q["flat_dust_pct"])
            moved_shape = met["shape_pct"] > float(q["flat_shape_pct"])
            verdict = ("poussières déplacées" if moved_dust else "") + (
                (" et " if moved_dust else "") + "forme différente (vignetage)" if moved_shape else "")
            row = {"filter": filt, "a": a.id, "b": b.id, "a_labels": a.folder_labels, "b_labels": b.folder_labels,
                   **met, "verdict": verdict or "identiques"}
            rows.append(row)
            ratio_path = workdir / "qa" / "flats" / f"ratio_{a.id}_vs_{b.id}.fits"
            fits.writeto(ratio_path, (flats[a.id] / flats[b.id]).astype(np.float32), overwrite=True)
            if not verdict:
                continue
            la = ", ".join(a.folder_labels) or f"{a.start:%d/%m}"
            lb = ", ".join(b.folder_labels) or f"{b.start:%d/%m}"
            shared = [g for g in inv.light_groups if g.filter == filt and g.flat and g.flat.sets
                      and (g.flat.reasons or g.flat.level == "B")
                      and g.flat.sets[0].id in (a.id, b.id)]
            sev = "warning" if moved_dust else "info"
            items.append(QAItem(
                "P4", sev,
                f"flats {filt} du {a.start:%d/%m} ({la}) et du {b.start:%d/%m} ({lb}) : {verdict} "
                f"(structure locale {met['dust_pct']:.2f} %, forme {met['shape_pct']:.2f} %, bruit {met['noise_pct']:.2f} %)"
                + (" — utilisés pour " + ", ".join(_group_label(g) for g in shared) if shared else ""),
                [str(ratio_path)],
                ("poussières déplacées : un flat d'une autre session laisse des anneaux → flux faux pour les "
                 "étoiles concernées" if moved_dust else
                 "forme du flat différente (souvent l'éclairage du panneau ou du ciel) : erreur lente avec la "
                 "position dans le champ, en partie absorbée par la photométrie différentielle (terme de champ)"),
                "faire des flats à chaque session sans démonter le train optique entre lights et flats ; "
                "nettoyer le filtre/capteur seulement juste avant une nouvelle série de flats", row))
    return items, rows


# =========================================================================== post-calibration checks


def calibrated_checks(plan: dict[str, Any], results: dict[str, Any], cfg: dict[str, Any],
                      step: int = 4) -> tuple[list[QAItem], list[dict[str, Any]]]:
    q = cfg["qa"]
    bg_min = float(q["calibrated_background_min_adu"])
    items: list[QAItem] = []
    rows: list[dict[str, Any]] = []
    for key, res in results.items():
        group = plan["groups"][key]
        zones: list[tuple[np.ndarray, np.ndarray]] = []
        if group.get("ampglow") and Path(group["ampglow"]).exists():
            from scipy import ndimage
            ag = fits.getdata(group["ampglow"])[::step, ::step] > 0
            lab, nz = ndimage.label(ag)
            for zi in range(1, nz + 1):
                z = lab == zi
                # Local comparison: the zone vs a 48-px band around it (a whole-field comparison would
                # mostly measure the sky/nebula gradient).
                band = ndimage.binary_dilation(z, iterations=max(1, 48 // step)) & ~ag
                if z.sum() >= 16 and band.sum() >= 16:
                    zones.append((z, band))
        per = []
        for raw, out in res["outputs"].items():
            with fits.open(out, memmap=False) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
            sub = data[::step, ::step]
            finite = np.isfinite(sub)
            nan = float(1 - finite.mean())
            vals = sub[finite]
            med = float(np.median(vals))
            sig = _rsig(vals)
            clip = vals[np.abs(vals - med) < 3 * sig] if sig > 0 else vals
            bg = float(np.median(clip))
            zero = float(np.mean(vals == 0))
            neg = float(np.mean(vals < bg - 5 * sig)) if sig > 0 else 0.0
            resid = [float(np.median(sub[z & finite]) - np.median(sub[b & finite]))
                     for z, b in zones if z.shape == sub.shape]
            per.append({"file": out, "raw": raw, "background_adu": round(bg, 2), "sigma_adu": round(sig, 2),
                        "frac_nan": nan, "frac_zero": zero, "frac_low_5sigma": neg,
                        "ampglow_zone_resid_adu": [round(r, 2) for r in resid]})
        if not per:
            continue
        bgs = [p["background_adu"] for p in per]
        row = {"group": key, "label": f"{group['label'] or group['session']} / {group['filter']}",
               "n": len(per), "background_min_adu": min(bgs), "background_median_adu": float(np.median(bgs)),
               "sigma_median_adu": float(np.median([p["sigma_adu"] for p in per])),
               "frac_nan_max": max(p["frac_nan"] for p in per), "frac_zero_max": max(p["frac_zero"] for p in per)}
        nzr = min((len(p["ampglow_zone_resid_adu"]) for p in per), default=0)
        zr = [float(np.median([p["ampglow_zone_resid_adu"][i] for p in per])) for i in range(nzr)]
        row["ampglow_zone_resid_adu"] = [round(r, 2) for r in zr]
        row["ampglow_excess_median_adu"] = max(zr, key=abs) if zr else None
        row["dark_id"] = (group.get("dark") or {}).get("id")
        row["session"] = group["session"]
        rows.append(row)
        lbl = row["label"]
        n_before = len(items)
        low = [p for p in per if p["background_adu"] < bg_min]
        if low:
            items.append(QAItem("C3", "error", f"{lbl} : fond calibré négatif sur {len(low)}/{len(per)} image(s) "
                                f"(min {min(bgs):.1f} ADU) → dark trop haut (offset ou dark d'une autre nuit)",
                                [p["raw"] for p in low], "flux faux, surtout pour les étoiles faibles",
                                "darks de la même nuit, offset fixé", {"background_min_adu": min(bgs)}))
        if row["frac_nan_max"] > 0 or row["frac_zero_max"] > 1e-3:
            items.append(QAItem("C3", "warning", f"{lbl} : pixels invalides (NaN jusqu'à "
                                f"{100 * row['frac_nan_max']:.3f} %, zéros jusqu'à {100 * row['frac_zero_max']:.3f} %)",
                                [], "mesures touchées à marquer", ""))
        for i in items[n_before:]:
            i.data["group"] = key
    items += _ampglow_residual_items(rows)
    return _sorted(items), rows


def _ampglow_residual_items(rows: list[dict[str, Any]], limit: float = 3.0) -> list[QAItem]:
    """C1: a dark residual is the same (in ADU) in every filter calibrated with that dark;
    a residual that changes with the filter comes from the sky or the nebula, not from the dark."""
    items: list[QAItem] = []
    by_dark: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("dark_id") and r.get("ampglow_zone_resid_adu"):
            by_dark[(r["session"], r["dark_id"])].append(r)
    for _, rs in by_dark.items():
        nz = min(len(r["ampglow_zone_resid_adu"]) for r in rs)
        for zi in range(nz):
            vals = [r["ampglow_zone_resid_adu"][zi] for r in rs]
            if max(abs(v) for v in vals) <= limit:
                continue
            labels = ", ".join(f"{r['label']} {r['ampglow_zone_resid_adu'][zi]:+.1f}" for r in rs)
            same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
            consistent = (len(rs) > 1 and same_sign and min(abs(v) for v in vals) > limit
                          and min(abs(v) for v in vals) / max(abs(v) for v in vals) >= 0.5)
            data = {"zone": zi + 1, "residuals_adu": vals, "group": rs[0]["group"]}
            if consistent:
                e = float(np.mean(vals))
                items.append(QAItem("C1", "warning", f"zone d'amp glow n° {zi + 1} : {e:+.1f} ADU par rapport au fond "
                                    f"voisin, identique dans tous les filtres ({labels}) → dark "
                                    f"{'sous' if e > 0 else 'sur'}-corrigé",
                                    [], "fond non plat près des bords → photométrie biaisée pour les étoiles concernées",
                                    "dark de même pose, même température et même nuit que les lights", data))
            else:
                why = ("différent selon le filtre → structure du ciel ou de la nébuleuse, pas le dark"
                       if len(rs) > 1 else "un seul filtre : impossible de séparer dark et ciel")
                items.append(QAItem("C1", "info", f"zone d'amp glow n° {zi + 1} : écart au fond voisin {labels} ADU ({why})",
                                    [], "", "", data))
    return items


# =========================================================================== report


def _md_item(i: QAItem) -> list[str]:
    L = [f"- **{i.check} — {SEVERITY_FR[i.severity]}** : {i.message}"]
    if i.impact:
        L.append(f"  - Impact : {i.impact}")
    if i.advice:
        L.append(f"  - Conseil pour la prochaine séance : {i.advice}")
    if i.files:
        shown = ", ".join(f"`{Path(f).name}`" for f in i.files[:3])
        more = f" (+{len(i.files) - 3})" if len(i.files) > 3 else ""
        L.append(f"  - Fichiers : {shown}{more}")
    return L


def summary_line(items: list[QAItem]) -> str:
    n = {s: sum(1 for i in items if i.severity == s) for s in SEVERITIES}
    return f"{n['error']} erreur(s), {n['warning']} avertissement(s), {n['info']} info(s)"


def items_markdown(items: list[QAItem], title: str = "## Contrôles qualité") -> list[str]:
    L = [title, "", f"_{summary_line(items)} — détail et méthode : `qa/qa.md`, docs/QA_CHECKS.md._", ""]
    if not items:
        return L + ["Aucun problème détecté.", ""]
    for i in items:
        L += _md_item(i)
    return L + [""]


def qa_markdown(report: dict[str, Any]) -> str:
    items = [QAItem(**d) for stage in ("header", "raw", "calibrated") for d in report.get(stage, [])]
    L = ["# Contrôles qualité", "", f"- Généré le {report.get('updated')}", f"- Bilan : {summary_line(items)}", ""]
    stages = {"header": "1. En-têtes", "raw": "2. Pixels des images brutes", "calibrated": "3. Après calibration"}
    for stage, title in stages.items():
        if stage not in report:
            continue
        si = [QAItem(**d) for d in report[stage]]
        L += [f"## {title}", ""]
        if not si:
            L += ["Aucun problème détecté.", ""]
        for i in si:
            L += _md_item(i)
        L.append("")
    t = report.get("tables", {})
    if t.get("dark_levels"):
        L += ["## Niveau des darks par nuit (contrôle P1)", "",
              "| Série | Pose | Nuit | N | Niveau ADU | σ ADU | T °C | Dossier(s) |", "|---|---|---|---|---|---|---|---|"]
        for r in t["dark_levels"]:
            L.append(f"| {r['set']} | {r['exptime']:g} s | {r['night']} | {r['n']} | {r['level_adu']:.1f} | "
                     f"{r['sigma_adu']:.2f} | {r['mean_ccd_temp'] if r['mean_ccd_temp'] is not None else '—'} | "
                     f"{', '.join(r['folders'])} |")
        L.append("")
    if t.get("raw_sky"):
        L += ["## Fond de ciel brut estimé (contrôle P8)", "", "Médiane des lights − niveau des darks appariés. "
              "Une valeur négative signifie des darks plus hauts que les lights.", "",
              "| Groupe | Lights ADU | Darks ADU | Fond estimé ADU | Niveau dark |", "|---|---|---|---|---|"]
        for r in t["raw_sky"]:
            L.append(f"| {r['group']} | {r['light_median_adu']:.1f} | {r['dark_level_adu']:.1f} | "
                     f"{r['raw_sky_adu']:+.1f} | {r['dark_level']} |")
        L.append("")
    if t.get("flat_pairs"):
        L += ["## Flats successifs (contrôle P4)", "", "Rapport de deux flats normalisés. *Structure locale* : "
              "anneaux de poussière déplacés ; *forme* : vignetage différent (caméra tournée, tirage, panneau). "
              "Les rapports sont dans `qa/flats/ratio_*.fits`.", "",
              "| Filtre | Flat 1 | Flat 2 | Structure locale % | Forme % | Bruit % | Verdict |",
              "|---|---|---|---|---|---|---|"]
        for r in t["flat_pairs"]:
            L.append(f"| {r['filter']} | {r['a']} ({', '.join(r['a_labels'])}) | {r['b']} ({', '.join(r['b_labels'])}) | "
                     f"{r['dust_pct']:.2f} | {r['shape_pct']:.2f} | {r['noise_pct']:.2f} | {r['verdict']} |")
        L.append("")
    if t.get("calibrated"):
        L += ["## Images calibrées (contrôles C1, C3)", "",
              "| Groupe | N | Fond min ADU | Fond médian ADU | σ ADU | Amp glow résiduel ADU (zone la plus forte) | NaN max | Zéros max |",
              "|---|---|---|---|---|---|---|---|"]
        for r in t["calibrated"]:
            ag = "—" if r["ampglow_excess_median_adu"] is None else f"{r['ampglow_excess_median_adu']:+.1f}"
            L.append(f"| {r['label']} | {r['n']} | {r['background_min_adu']:.1f} | {r['background_median_adu']:.1f} | "
                     f"{r['sigma_median_adu']:.2f} | {ag} | {100 * r['frac_nan_max']:.3f} % | "
                     f"{100 * r['frac_zero_max']:.3f} % |")
        L.append("")
    return "\n".join(L) + "\n"


def load_report(workdir: Path) -> dict[str, Any]:
    p = workdir / "qa" / "qa.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save_report(workdir: Path, report: dict[str, Any]) -> None:
    report["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    d = workdir / "qa"
    d.mkdir(parents=True, exist_ok=True)
    (d / "qa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (d / "qa.md").write_text(qa_markdown(report), encoding="utf-8")


def run_checks(inv: Inventory, cfg: dict[str, Any], workdir: Path, raw: bool = True) -> dict[str, Any]:
    """Header (+ raw pixel) checks; writes qa/qa.json and qa/qa.md. Calibrated checks are kept."""
    report = load_report(workdir)
    report["header"] = [i.to_dict() for i in header_checks(inv, cfg)]
    if raw:
        items, tables = raw_checks(inv, cfg, workdir)
        report["raw"] = [i.to_dict() for i in items]
        report.setdefault("tables", {}).update(tables)
    save_report(workdir, report)
    for d in report["header"] + report.get("raw", []):
        getattr(log, {"error": "error", "warning": "warning"}.get(d["severity"], "info"))(
            f"[{d['check']}] {d['message']}")
    return report


def all_items(report: dict[str, Any], stages: Iterable[str] = ("header", "raw", "calibrated")) -> list[QAItem]:
    return _sorted(QAItem(**d) for s in stages for d in report.get(s, []))
