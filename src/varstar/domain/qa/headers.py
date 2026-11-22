"""Header checks (H*): from the inventory only, no pixel is read."""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections import defaultdict
from typing import Any

from ..model import DARK, DARKFLAT, FLAT, LIGHT, Frame, Inventory, LightGroup
from .model import QAItem, group_label, night_label, offset_hours, sort_items


def header_checks(inv: Inventory, cfg: dict[str, Any]) -> list[QAItem]:
    q = cfg["qa"]
    items: list[QAItem] = []
    off = offset_hours(cfg)

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
            f"{kind_fr}, nuit du {night_label(inv, night)} : {len(bad)}/{len(frames)} pose(s) à plus de "
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
            mismatched.append(f"{group_label(g)} (lights {', '.join(sorted(lv))}, darks {', '.join(sorted(cv))})")
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
            f"{group_label(g)} : flats pris à FOCPOS {fmed:.0f}, lights entre {lo:.0f} et {hi:.0f} "
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
                        f"{group_label(g)} ({obj}) : {len(far)}/{g.n} pose(s) pointée(s) à plus de {limit:.0f}′ du "
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
            items.append(QAItem("H7", "info", f"nuit du {night_label(inv, night)} : couverture nuageuse annoncée "
                                f"> {cmax:g} % sur {len(cloudy)}/{len(frames)} pose(s) (source météo souvent en ligne : "
                                "indicatif)", [f.rel_path for f in cloudy],
                                "extinction variable : à confirmer par le point zéro (contrôle qualité M3)", ""))
        if dew:
            mn = min(f.ambtemp - f.dewpoint for f in dew)
            items.append(QAItem("H7", "warning", f"nuit du {night_label(inv, night)} : température à moins de "
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
            items.append(QAItem("H9", "info", f"{group_label(g)} : {g.dark.n_frames} darks seulement "
                                f"(< {q['min_darks']}) pour le master", [],
                                "master dark plus bruité, rejet des cosmiques moins robuste",
                                f"faire au moins {q['min_darks']} darks par pose et par nuit"))
        if g.flat and g.flat.sets and g.flat.n_frames < int(q["min_flats"]):
            items.append(QAItem("H9", "info", f"{group_label(g)} : {g.flat.n_frames} flats seulement "
                                f"(< {q['min_flats']})", [], "flat plus bruité",
                                f"faire au moins {q['min_flats']} flats par filtre"))
    return sort_items(items)
