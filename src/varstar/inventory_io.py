"""Serialisation of the inventory: inventory.json (machine) and inventory.md (French, human-readable)."""

from __future__ import annotations

import datetime as dt
from typing import Any

from .domain.model import DARK, DARKFLAT, FLAT, LIGHT, DarkTodo, FrameSet, Inventory, LightGroup, Match
from .inventory import darks_todo

SEVERITY_FR = {"error": "ERREUR", "warning": "Avertissement", "info": "Info"}


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


# --------------------------------------------------------------------------- markdown


def _fmt_t(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}"


def _fmt_match(m: Match | None) -> str:
    if m is None:
        return "—"
    if not m.sets:
        return f"**{m.level}** aucun"
    if len(m.sets) > 1:
        exps = "/".join(sorted({f"{s.exptime:g}" for s in m.sets}))
        return f"**{m.level}** {m.n_frames} × {exps} s ({len(m.sets)} séries)"
    s = m.sets[0]
    return f"**{m.level}** {s.n} × {s.exptime:g} s ({s.start:%d/%m})"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return out


def todo_markdown(todos: list[DarkTodo]) -> list[str]:
    if not todos:
        return ["Aucun dark manquant : toutes les sessions ont des darks et darks de flat de niveau A."]
    rows = []
    for t in todos:
        kind = "dark (lights)" if t.kind == "dark" else "dark de flat"
        rows.append([kind, f"{t.exptime:g} s", f"{t.sensor_key.gain:g}" if t.sensor_key.gain is not None else "?",
                     f"{t.sensor_key.xbin}x{t.sensor_key.ybin}", _fmt_t(t.set_temp), _fmt_t(t.mean_temp),
                     ", ".join(t.filters), "<br>".join(t.sessions), t.current_level, t.reason])
    return _md_table(["Type", "Pose", "Gain", "Binning", "Consigne °C", "T moy. °C", "Filtres",
                      "Sessions", "Niveau actuel", "Raison"], rows)


def inventory_markdown(inv: Inventory, qa_lines: list[str] | None = None) -> str:
    counts = inv.counts_by_type()
    levels = inv.levels_by_session_filter()
    lines: list[str] = [
        "# Inventaire varstar",
        "",
        f"- Dossier source : `{inv.root}`",
        f"- Généré le : {inv.created:%d/%m/%Y %H:%M}",
        f"- Fichiers FITS retenus : {sum(counts.values())} — "
        + ", ".join(f"{k} {counts.get(k, 0)}" for k in (LIGHT, DARK, FLAT, DARKFLAT, "BIAS")),
        "",
        "Les fichiers sont classés **d'après leurs en-têtes FITS** ; les noms de dossiers servent "
        "seulement d'étiquettes. Session = nuit d'observation (DATE-OBS UTC − 12 h).",
        "",
        *(qa_lines or []),
        "## Niveaux de calibration par session et filtre",
        "",
    ]
    sessions = sorted({k[0] for k in levels})
    filters = sorted({k[1] for k in levels})
    rows = [[inv.session_label(s)] + [levels.get((s, f), "·") for f in filters] for s in sessions]
    lines += _md_table(["Session"] + filters, rows)
    lines += ["", "A : calibration complète · B : repli journalisé · C : calibration partielle "
              "(sans flats → courbes valables intra-session uniquement)", ""]

    lines += ["## Détail des groupes de lights", ""]
    rows = []
    for g in inv.light_groups:
        rows.append([inv.session_label(g.session), g.filter, f"{g.n} × {g.exptime:g} s",
                     f"{g.start:%d/%m %H:%M}–{g.end:%H:%M}", _fmt_t(g.mean_temp),
                     _fmt_match(g.dark), _fmt_match(g.flat), _fmt_match(g.darkflat), f"**{g.level}**",
                     "<br>".join(g.reasons) or "—"])
    lines += _md_table(["Session", "Filtre", "Lights", "UTC", "T moy. °C", "Darks", "Flats",
                        "Darks de flat", "Niveau", "Remarques"], rows)

    lines += ["", "## Capteur et unités", ""]
    rows = []
    for s in inv.sensors:
        rows.append([s.key.label(), s.n_frames, f"{s.egain:.4f}" if s.egain else "—",
                     f"×{s.adu_scale}" if s.adu_scale else "non mesuré",
                     f"{s.effective_gain:.4f}" if s.effective_gain else "—",
                     s.saturation_adu or "—",
                     f"{s.pixel_scale_arcsec:.2f}" if s.pixel_scale_arcsec else "—"])
    lines += _md_table(["Configuration capteur", "Fichiers", "EGAIN (en-tête)", "Facteur ADU détecté",
                        "Gain effectif e⁻/ADU", "Saturation ADU", "Échelle ″/px"], rows)
    lines += ["", "Facteur ADU détecté sur les pixels d'une image (valeurs toutes multiples du facteur). "
              "Échelle calculée avec XPIXSZ tel qu'écrit par N.I.N.A. (déjà multiplié par le binning).", ""]

    lines += ["## Séries de calibration", ""]
    for kind, title in ((DARK, "Darks"), (FLAT, "Flats"), (DARKFLAT, "Darks de flat")):
        sets = inv.calibration_sets.get(kind, [])
        lines += [f"### {title}", ""]
        if not sets:
            lines += ["Aucune.", ""]
            continue
        used_by: dict[str, set[str]] = {}
        for g in inv.light_groups:
            for m in (g.dark, g.flat, g.darkflat):
                for s in (m.sets if m else []):
                    used_by.setdefault(s.id, set()).add(f"{g.label or g.session}/{g.filter}")
        rows = [[s.id, s.filter or "—", f"{s.exptime:g} s", s.n, f"{s.start:%d/%m %H:%M}–{s.end:%d/%m %H:%M}",
                 _fmt_t(s.mean_temp), "<br>".join(s.folders), ", ".join(sorted(used_by.get(s.id, []))) or "—"]
                for s in sets]
        lines += _md_table(["Série", "Filtre", "Pose", "N", "UTC", "T moy. °C", "Dossiers", "Utilisée par"], rows)
        lines.append("")

    lines += ["## Anomalies", ""]
    if not inv.anomalies:
        lines.append("Aucune.")
    for a in sorted(inv.anomalies, key=lambda a: ("error", "warning", "info").index(a.severity)):
        lines.append(f"- **{SEVERITY_FR[a.severity]}** — {a.message}")
        if a.files and len(a.files) <= 6 and a.code not in ("unused_sets",):
            for f in a.files:
                lines.append(f"  - `{f}`")
        elif a.files:
            lines.append(f"  - ({len(a.files)} éléments, liste complète dans inventory.json)")
    lines += ["", "## Darks à refaire", ""]
    lines += todo_markdown(darks_todo(inv))
    lines.append("")
    return "\n".join(lines)
