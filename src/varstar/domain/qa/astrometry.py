"""Astrometry checks (A*): what the WCS solutions reveal about the acquisition.

Input: one row per calibrated image (see ``application.astrometry``), with the group's
session, filter, flat master and the solution summary. Pure: no file access.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..astrometry import angle_diff_deg, angular_sep_arcsec, camera_angle_deg, circular_median_deg
from ..settings import AstrometrySettings
from .model import QAItem, sort_items


def _label(g: Mapping[str, Any]) -> str:
    return f"{g.get('label') or g['session']} / {g['filter']}"


def astrometry_checks(groups: Sequence[Mapping[str, Any]], a: AstrometrySettings,
                      max_field_frac: float) -> list[QAItem]:
    """``groups``: {key, session, label, filter, flat_id, expected_scale, focallen, xpixsz, images: [row]};
    a row: {raw, ok, reason, scale_arcsec, orientation_deg, center: [ra, dec], segment, time, field_deg}."""
    items: list[QAItem] = []
    solved = [(g, r) for g in groups for r in g["images"] if r["ok"]]

    # A1: images not solved.
    for g in groups:
        bad = [r for r in g["images"] if not r["ok"]]
        if not bad:
            continue
        reasons = Counter(r["reason"] or "?" for r in bad)
        every = len(bad) == len(g["images"])
        items.append(QAItem(
            "A1", "error" if every else "warning",
            f"{_label(g)} : {len(bad)}/{len(g['images'])} image(s) non résolue(s) — "
            + " ; ".join(f"{why} ({n})" for why, n in reasons.most_common()),
            [r["raw"] for r in bad], "images exclues de la photométrie",
            "vérifier la mise au point, les nuages et la buée ; RA/DEC, FOCALLEN et XPIXSZ doivent être "
            "écrits dans les en-têtes (N.I.N.A. : monture et télescope connectés)",
            {"group": g["key"], "n_failed": len(bad)}))

    # A2: true pixel scale vs header (focal length).
    by_exp: dict[tuple, list[float]] = defaultdict(list)
    for g, r in solved:
        if g.get("expected_scale"):
            by_exp[(g["expected_scale"], g.get("focallen"), g.get("xpixsz"))].append(r["scale_arcsec"])
    for (expected, focallen, xpixsz), scales in by_exp.items():
        real = float(np.median(scales))
        pct = 100 * (real / expected - 1)
        if abs(pct) > a.scale_check_pct:
            f_real = 206.264806 * xpixsz / real if xpixsz else None
            items.append(QAItem(
                "A2", "info",
                f"échelle réelle {real:.3f}″/px contre {expected:.3f}″/px attendus d'après l'en-tête ({pct:+.1f} %)"
                + (f" → focale effective ≈ {f_real:.0f} mm (FOCALLEN = {focallen:g} mm)" if f_real and focallen else ""),
                [], "aucun (la WCS mesure l'échelle réelle) ; l'échelle de l'en-tête sert seulement d'indice",
                "renseigner la focale réelle (réducteur, correcteur) dans le profil N.I.N.A.",
                {"scale_arcsec": real, "expected_arcsec": expected}))

    # A3: framing differs between groups (true centres).
    if solved:
        centers = np.array([r["center"] for _, r in solved])
        ra0, dec0 = circular_median_deg(centers[:, 0]), float(np.median(centers[:, 1]))
        for g in groups:
            rows = [r for r in g["images"] if r["ok"]]
            if not rows:
                continue
            c = np.array([r["center"] for r in rows])
            gra, gdec = circular_median_deg(c[:, 0]), float(np.median(c[:, 1]))
            off = float(angular_sep_arcsec(gra, gdec, ra0, dec0)) / 60
            field = min(rows[0]["field_deg"]) * 60
            if off > max_field_frac * field:
                items.append(QAItem(
                    "A3", "warning",
                    f"{_label(g)} : champ décalé de {off:.0f}′ par rapport au cadrage habituel (champ {field:.0f}′)",
                    [], "étoiles de comparaison différentes, bords du champ (vignetage, distorsion)",
                    "recentrer par résolution astrométrique (« Center » dans N.I.N.A.) sur les mêmes coordonnées "
                    "à chaque session", {"group": g["key"], "offset_arcmin": round(off, 1)}))

    # A4: meridian flips (several segments in a night).
    by_session: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for g, r in solved:
        by_session[g["session"]].append((g, r))
    for session, pairs in sorted(by_session.items()):
        segs = sorted({r["segment"] for _, r in pairs})
        if len(segs) > 1:
            parts = []
            for sg in segs:
                rows = [r for _, r in pairs if r["segment"] == sg]
                times = sorted(r["time"] or "" for r in rows)
                orient = circular_median_deg(r["orientation_deg"] for r in rows)
                parts.append(f"segment {sg} : {len(rows)} image(s), orientation {orient:.0f}°, "
                             f"{times[0][11:16]}–{times[-1][11:16]} TU")
            labels = sorted({g.get("label") or session for g, _ in pairs})
            items.append(QAItem(
                "A4", "info", f"nuit du {session} ({', '.join(labels)}) : {len(segs)} segments (retournement au "
                f"méridien) — " + " ; ".join(parts), [],
                "la photométrie traite chaque segment à part (point zéro par segment)", "",
                {"session": session, "segments": segs}))

    # A5: camera rotation between sessions (orientation modulo 180°, flips ignored).
    cam: dict[str, float] = {}
    for session, pairs in by_session.items():
        cam[session] = circular_median_deg((camera_angle_deg(r["orientation_deg"]) for _, r in pairs), 180.0)
    if len(cam) > 1:
        ref = circular_median_deg(cam.values(), 180.0)
        moved = {s: v for s, v in cam.items()
                 if min(angle_diff_deg(v, ref), 180 - angle_diff_deg(v, ref)) > a.camera_rotation_tol_deg}
        if moved:
            items.append(QAItem(
                "A5", "info", "angle de la caméra par nuit : " + " ; ".join(
                    f"{s} {v:.1f}°" for s, v in sorted(cam.items())) + f" (référence {ref:.1f}°)", [],
                "angle mesuré par rapport au ciel : sans effet sur les flats si toute l'optique tourne avec la "
                "caméra ; un flat ne vaut plus si la caméra seule a tourné (rotateur, démontage)", "",
                {"camera_angle_deg": cam}))
        items += _borrowed_flat_rotations(groups, cam, a)
    return sort_items(items)


def _rotation_answer(night: str, a: AstrometrySettings) -> bool | None:
    answers = {d.isoformat(): v for d, v in a.whole_train_rotation_nights}
    return answers.get(night, a.whole_train_rotation)


def _borrowed_flat_rotations(groups: Sequence[Mapping[str, Any]], cam: Mapping[str, float],
                             a: AstrometrySettings) -> list[QAItem]:
    """A night calibrated with the flat of another night, at another camera angle on the sky.

    The astrometry cannot tell what turned: the whole optical train (flat still valid) or the
    camera alone (dust and vignetting moved: flat wrong). The answer is the user's
    (``astrometry.whole_train_rotation[_nights]``); without it, the question is asked.
    """
    def diff(x: float, y: float) -> float:
        d = angle_diff_deg(x, y)
        return min(d, 180 - d)

    items: list[QAItem] = []
    by_flat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for g in groups:
        if g.get("flat_id") and g["session"] in cam:
            by_flat[g["flat_id"]].append(g)
    for flat_id, gs in by_flat.items():
        owners = sorted({g["session"] for g in gs if not g.get("flat_borrowed")})
        borrowers = sorted({g["session"] for g in gs if g.get("flat_borrowed")})
        if not borrowers or not owners:
            continue
        ref = circular_median_deg((cam[s] for s in owners), 180.0)
        for night in borrowers:
            spread = diff(cam[night], ref)
            if spread <= a.camera_rotation_tol_deg:
                continue
            labels = ", ".join(sorted(_label(g) for g in gs if g["session"] == night))
            facts = (f"{labels} : flat de la nuit du {', '.join(owners)} pris à un autre angle de caméra sur le ciel "
                     f"({ref:.1f}° contre {cam[night]:.1f}°, écart {spread:.1f}°)")
            data = {"flat_id": flat_id, "night": night, "flat_nights": owners, "spread_deg": round(spread, 2)}
            answer = _rotation_answer(night, a)
            if answer is True:
                items.append(QAItem("A5", "info", facts + " — tout le train optique a tourné (réponse donnée) : "
                                    "flat valable", [], "", "", {**data, "answer": True}))
            elif answer is False:
                items.append(QAItem(
                    "A5", "error", facts + " — la caméra a tourné seule (réponse donnée) : flat inadapté",
                    [], "poussières et vignetage décalés : erreurs de flux selon la position dans le champ",
                    f"retirer la nuit du {night} de matching.same_optical_setup (le groupe passera au niveau C, "
                    "courbes valables dans la nuit seulement) ou refaire des flats après chaque rotation de la caméra",
                    {**data, "answer": False}))
            else:
                items.append(QAItem(
                    "A5", "warning", facts + ". QUESTION : qu'est-ce qui a tourné entre ces nuits ? Tout le train "
                    "optique (tube et caméra ensemble : le flat reste valable) ou la caméra seule (rotateur, "
                    "démontage : le flat ne convient plus) ?", [],
                    "inconnu tant que la question n'a pas de réponse : si la caméra a tourné seule, flux faux "
                    "selon la position dans le champ",
                    f"répondre dans la configuration : astrometry.whole_train_rotation_nights: {{{night}: true}} "
                    "(tout le train optique) ou false (caméra seule) ; astrometry.whole_train_rotation pour toutes "
                    "les nuits", {**data, "question": True}))
    return items


def field_center(solution, width: int, height: int) -> tuple[float, float]:
    ra, dec = solution.pixel_to_world((width - 1) / 2, (height - 1) / 2)
    return float(ra), float(dec)


__all__ = ["astrometry_checks", "field_center"]
