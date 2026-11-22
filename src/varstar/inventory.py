"""Inventory: sessions, calibration sets, light <-> dark/flat/dark-flat matching, A/B/C levels.

Everything here works from FITS headers only. Folder names are used as labels
(SESSION_xxx) and for consistency warnings, never for classification.
"""

from __future__ import annotations

import datetime as dt
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .adapters.fits_io import detect_adu_scale_file, read_frame
from .config import optical_setup_periods
from .discovery import discover
from .domain.model import (BIAS, DARK, DARKFLAT, FLAT, LEVELS, LIGHT, UNKNOWN, Anomaly,  # noqa: F401
                           DarkTodo, Frame, FrameSet, Inventory, LightGroup, Match, SensorInfo, SensorKey,
                           worst)

# --------------------------------------------------------------------------- helpers


def _fmt_exp(x: float) -> str:
    return f"{x:g}s"


def _cluster(frames: list[Frame], gap_hours: float) -> list[list[Frame]]:
    ordered = sorted(frames, key=lambda f: f.date_obs)
    clusters: list[list[Frame]] = []
    for f in ordered:
        if clusters and (f.date_obs - clusters[-1][-1].date_obs) <= dt.timedelta(hours=gap_hours):
            clusters[-1].append(f)
        else:
            clusters.append([f])
    return clusters


def _exp_bucket(exptime: float) -> float:
    return round(exptime, 2)


def build_sets(frames: Iterable[Frame], kind: str, gap_hours: float) -> list[FrameSet]:
    groups: dict[tuple, list[Frame]] = defaultdict(list)
    for f in frames:
        filt = f.filter if kind == FLAT else None
        groups[(f.sensor_key, _exp_bucket(f.exptime), filt)].append(f)
    sets: list[FrameSet] = []
    for (key, exp, filt), members in groups.items():
        for cluster in _cluster(members, gap_hours):
            sets.append(FrameSet("", kind, key, exp, filt, cluster))
    sets.sort(key=lambda s: (s.start, s.exptime, s.filter or ""))
    counters: Counter = Counter()
    for s in sets:
        base = f"{kind}-{s.filter + '-' if s.filter else ''}{_fmt_exp(s.exptime)}-{s.start:%Y%m%d}"
        counters[base] += 1
        s.id = base if counters[base] == 1 else f"{base}-{counters[base]}"
    # Disambiguate the first one if a duplicate appeared later.
    dup = {b for b, c in counters.items() if c > 1}
    for s in sets:
        if s.id in dup:
            s.id = f"{s.id}-1"
    return sets


def _interval_gap(a0: dt.datetime, a1: dt.datetime, b0: dt.datetime, b1: dt.datetime) -> dt.timedelta:
    """Time separating two intervals (zero if they overlap)."""
    if b0 > a1:
        return b0 - a1
    if a0 > b1:
        return a0 - b1
    return dt.timedelta(0)


def _in_same_period(d1: dt.date, d2: dt.date, periods: list[tuple[dt.date, dt.date]]) -> bool:
    return any(p0 <= d1 <= p1 and p0 <= d2 <= p1 for p0, p1 in periods)


# --------------------------------------------------------------------------- matching


def set_night(s: FrameSet, offset_hours: float = -12.0) -> dt.date:
    """Observing night of a calibration series (same convention as the light sessions)."""
    return (s.mid + dt.timedelta(hours=offset_hours)).date()


def _by_night(sets: list[FrameSet], offset_hours: float) -> dict[dt.date, list[FrameSet]]:
    out: dict[dt.date, list[FrameSet]] = defaultdict(list)
    for s in sets:
        out[set_night(s, offset_hours)].append(s)
    return out


def _nearest_night(sets: list[FrameSet], t0: dt.datetime, t1: dt.datetime,
                   offset_hours: float) -> tuple[dt.date, list[FrameSet], dt.timedelta]:
    """Sets of the single night closest in time to [t0, t1] (darks of two nights are never mixed)."""
    nights = _by_night(sets, offset_hours)

    def gap(ss: list[FrameSet]) -> dt.timedelta:
        return min(_interval_gap(t0, t1, s.start, s.end) for s in ss)

    night = min(nights, key=lambda n: (gap(nights[n]), n))
    return night, nights[night], gap(nights[night])


def match_darks(group: LightGroup, dark_sets: list[FrameSet], cfg: dict[str, Any]) -> Match:
    """Darks for a light group.

    Policy (D16): darks are taken from the same night as the lights only. Darks of different
    nights are never combined, because the sensor offset can change from one night to the next
    without any header trace (2021-09-06: +25 ADU). Without same-night darks, the darks of the
    nearest night are used at level B (offset not verifiable, checked after calibration).
    """
    m = cfg["matching"]
    tol, ttol = m["exptime_tolerance_s"], m["temp_tolerance_c"]
    same_night_only = bool(m.get("dark_same_night", True))
    offset = float(cfg["inventory"].get("session_day_offset_hours", -12))
    same_key = [s for s in dark_sets if s.sensor_key == group.sensor_key]
    same_exp = [s for s in same_key if abs(s.exptime - group.exptime) <= tol]
    gt = group.mean_temp

    def dtemp(s: FrameSet) -> float:
        if gt is None or s.mean_temp is None:
            return 0.0
        return abs(s.mean_temp - gt)

    def temp_note(sets: list[FrameSet]) -> list[str]:
        if gt is None or any(s.mean_temp is None for s in sets):
            return ["darks : température CCD absente, tolérance non vérifiée"]
        return []

    ok = [s for s in same_exp if dtemp(s) <= ttol]
    if ok and not same_night_only:
        return Match("A", ok, temp_note(ok))
    if ok:
        tonight = [s for s in ok if set_night(s, offset) == group.session]
        if tonight:
            return Match("A", tonight, temp_note(tonight))
        night, chosen, gap = _nearest_night(ok, group.start, group.end, offset)
        return Match("B", chosen, temp_note(chosen) + [
            f"pas de dark de {_fmt_exp(group.exptime)} pris la même nuit : darks de la nuit du "
            f"{night:%d/%m} (écart {gap.total_seconds() / 86400:.1f} j) — offset non vérifiable, "
            "contrôlé après calibration"], borrowed_night=True)
    if same_exp:
        tonight = [s for s in same_exp if set_night(s, offset) == group.session]
        best = min(tonight or same_exp, key=lambda s: (dtemp(s), abs((s.mid - group.start).total_seconds())))
        return Match("B", [best], [
            f"darks {_fmt_exp(group.exptime)} : ΔT = {dtemp(best):.1f} °C > {ttol:g} °C (le plus proche retenu)"])
    if same_key and m.get("allow_dark_scaling", True):
        temp_ok = [s for s in same_key if dtemp(s) <= ttol] or same_key
        if same_night_only:
            tonight = [s for s in temp_ok if set_night(s, offset) == group.session]
            temp_ok = tonight or _nearest_night(temp_ok, group.start, group.end, offset)[1]
        shorter = [s for s in temp_ok if s.exptime < group.exptime]
        pool = shorter or temp_ok
        best_exp = max(s.exptime for s in pool) if shorter else min(
            (s.exptime for s in pool), key=lambda e: abs(e - group.exptime))
        chosen = [s for s in pool if s.exptime == best_exp]
        night = set_night(chosen[0], offset)
        where = "de la même nuit" if night == group.session else f"de la nuit du {night:%d/%m}"
        return Match("B", chosen, [
            f"pas de dark de {_fmt_exp(group.exptime)} : mise à l'échelle depuis des darks de "
            f"{_fmt_exp(best_exp)} {where} (repli)"])
    if same_key:
        return Match("C", [], [f"pas de dark de {_fmt_exp(group.exptime)} et mise à l'échelle désactivée "
                               "→ masque de pixels chauds seulement"])
    return Match("C", [], ["aucun dark compatible (clé capteur) → masque de pixels chauds seulement"])


def match_flats(group: LightGroup, session_span: tuple[dt.datetime, dt.datetime],
                flat_sets: list[FrameSet], cfg: dict[str, Any]) -> Match:
    m = cfg["matching"]
    same_hours = float(m.get("flat_same_session_hours", 16))
    max_age = dt.timedelta(days=float(m.get("max_flat_age_days", 7)))
    periods = optical_setup_periods(cfg)
    offset = float(cfg["inventory"].get("session_day_offset_hours", -12))
    s0, s1 = session_span

    cands = [s for s in flat_sets if s.sensor_key == group.sensor_key and s.filter == group.filter]
    if not cands:
        return Match("C", [], [f"aucun flat {group.filter} → courbes valables intra-session uniquement"])

    def gap(s: FrameSet) -> dt.timedelta:
        return _interval_gap(s0, s1, s.start, s.end)

    same = [s for s in cands if gap(s) <= dt.timedelta(hours=same_hours)]
    if same:
        return Match("A", [min(same, key=gap)], [])

    if m.get("trust_session_folder_flats", True) and group.label:
        labelled = [s for s in cands if group.label in s.folder_labels and gap(s) <= max_age]
        if labelled:
            best = min(labelled, key=gap)
            days = gap(best).total_seconds() / 86400
            when = "avant" if best.end < s0 else "après"
            return Match("A", [best], [
                f"flats {group.filter} du dossier SESSION_{group.label} pris le {best.start:%d/%m}, "
                f"{days:.1f} j {when} les lights (montage supposé inchangé, question Q3)"])

    session_date = group.session
    neighbours = [s for s in cands if gap(s) <= max_age and _in_same_period(
        session_date, (s.mid + dt.timedelta(hours=offset)).date(), periods)]
    if neighbours:
        best = min(neighbours, key=gap)
        days = gap(best).total_seconds() / 86400
        return Match("B", [best], [
            f"flats {group.filter} d'une session voisine ({best.start:%d/%m}, écart {days:.1f} j), "
            "même montage déclaré"])

    near = [s for s in cands if gap(s) <= max_age]
    if near:
        best = min(near, key=gap)
        days = gap(best).total_seconds() / 86400
        return Match("C", [], [
            f"aucun flat {group.filter} de la session ; flats du {best.start:%d/%m} (écart {days:.1f} j) "
            "non utilisés car le même montage n'est pas confirmé (same_optical_setup)"])
    return Match("C", [], [f"aucun flat {group.filter} à moins de {max_age.days} j"
                           " → courbes valables intra-session uniquement"])


def match_darkflats(flat_set: FrameSet, darkflat_sets: list[FrameSet], dark_sets: list[FrameSet],
                    cfg: dict[str, Any]) -> Match:
    tol = cfg["matching"]["darkflat_exptime_tolerance_s"]
    df = [s for s in darkflat_sets if s.sensor_key == flat_set.sensor_key]
    dk = [s for s in dark_sets if s.sensor_key == flat_set.sensor_key]
    exact = [s for s in df if abs(s.exptime - flat_set.exptime) <= tol]
    if not exact:
        exact = [s for s in dk if abs(s.exptime - flat_set.exptime) <= tol]
    if exact:
        if cfg["matching"].get("dark_same_night", True):
            offset = float(cfg["inventory"].get("session_day_offset_hours", -12))
            exact = _nearest_night(exact, flat_set.start, flat_set.end, offset)[1]
        return Match("A", exact, [])
    pool = df or dk
    if pool:
        best_exp = min((s.exptime for s in pool), key=lambda e: abs(e - flat_set.exptime))
        chosen = [s for s in pool if s.exptime == best_exp]
        return Match("B", chosen, [
            f"pas de dark de flat de {_fmt_exp(flat_set.exptime)} (flats {flat_set.filter}) : "
            f"le plus proche ({_fmt_exp(best_exp)}) est utilisé"])
    return Match("C", [], [f"aucun dark de flat pour les flats {flat_set.filter} "
                           f"de {_fmt_exp(flat_set.exptime)} → flats inutilisables"])


# --------------------------------------------------------------------------- consistency checks

_EXP_DIR_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*s(?:ec)?$", re.IGNORECASE)
_TYPE_DIRS = {
    "light": {LIGHT}, "lights": {LIGHT},
    "dark": {DARK, DARKFLAT}, "darks": {DARK, DARKFLAT},
    "flat": {FLAT}, "flats": {FLAT},
    "bias": {BIAS}, "offset": {BIAS}, "offsets": {BIAS},
    "darkflat": {DARKFLAT, DARK}, "darkflats": {DARKFLAT, DARK},
    "flatdark": {DARKFLAT, DARK}, "flatdarks": {DARKFLAT, DARK},
}


def folder_consistency(frame: Frame, known_filters: set[str]) -> list[tuple[str, str]]:
    """Return (code, message) pairs where the folder path contradicts the header."""
    issues = []
    parts = Path(frame.rel_path).parent.parts
    for part in parts:
        m = _EXP_DIR_RE.match(part)
        if m and frame.exptime is not None:
            folder_exp = float(m.group(1).replace(",", "."))
            if abs(folder_exp - frame.exptime) > 0.5:
                issues.append(("folder_exptime",
                               f"pose {frame.exptime:g} s dans un dossier « {part} » → classé selon l'en-tête"))
    for part in parts:
        expected = _TYPE_DIRS.get(part.lower())
        if expected and frame.imagetyp not in expected and frame.imagetyp != UNKNOWN:
            issues.append(("folder_type",
                           f"type {frame.imagetyp} dans un dossier « {part} » → classé selon l'en-tête"))
            break
    if frame.imagetyp in (LIGHT, FLAT) and frame.filter:
        for part in parts:
            if part in known_filters and part != frame.filter:
                issues.append(("folder_filter",
                               f"filtre {frame.filter} dans un dossier « {part} » → classé selon l'en-tête"))
                break
    return issues


# --------------------------------------------------------------------------- main entry


def _sensor_infos(frames: list[Frame], read_pixels: bool, anomalies: list[Anomaly]) -> list[SensorInfo]:
    by_key: dict[SensorKey, list[Frame]] = defaultdict(list)
    for f in frames:
        by_key[f.sensor_key].append(f)
    infos = []
    for key, members in by_key.items():
        egain_values = [f.egain for f in members if f.egain is not None]
        egains = sorted({round(e, 6) for e in egain_values})
        if len(egains) > 1:
            anomalies.append(Anomaly("warning", "egain_varies",
                                     f"EGAIN varie pour {key.label()} : {egains}"))
        lights = [f for f in members if f.imagetyp == LIGHT] or members
        scale, source = None, None
        if read_pixels:
            sample = sorted(lights, key=lambda f: f.rel_path)[0]
            try:
                scale = detect_adu_scale_file(sample.path)
                source = sample.rel_path
            except Exception as exc:  # unreadable pixels must not stop the inventory
                anomalies.append(Anomaly("warning", "adu_scale_failed",
                                         f"détection du facteur ADU impossible : {exc}", [sample.rel_path]))
        xp = [f.xpixsz for f in members if f.xpixsz]
        fl = [f.focallen for f in members if f.focallen]
        egain = statistics.median(egain_values) if egain_values else None
        infos.append(SensorInfo(key, len(members), egain, scale, source,
                                statistics.median(xp) if xp else None,
                                statistics.median(fl) if fl else None))
    infos.sort(key=lambda s: -s.n_frames)
    return infos


def build_inventory(root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> Inventory:
    root = root.resolve()
    inv_cfg = cfg["inventory"]
    offset = float(inv_cfg.get("session_day_offset_hours", -12))
    gap_hours = float(inv_cfg.get("set_gap_hours", 3))
    anomalies: list[Anomaly] = []

    disc = discover(root, cfg, exclude=exclude)
    frames: list[Frame] = []
    unreadable = []
    for path in disc.fits_files:
        try:
            frames.append(read_frame(path, root))
        except Exception as exc:
            unreadable.append(f"{path.relative_to(root)} ({exc.__class__.__name__})")
    if unreadable:
        anomalies.append(Anomaly("error", "unreadable", f"{len(unreadable)} fichier(s) FITS illisible(s)",
                                 unreadable))

    # Frames with blocking header problems are excluded (and reported).
    usable: list[Frame] = []
    bad: dict[str, list[str]] = defaultdict(list)
    for f in frames:
        blocking = [i for i in f.issues]
        if blocking:
            for i in blocking:
                bad[i].append(f.rel_path)
        else:
            usable.append(f)
    for msg, files in bad.items():
        anomalies.append(Anomaly("warning", "bad_header", f"{msg} — fichier(s) exclu(s)", files))

    # Folder vs header consistency (aggregated per folder and message).
    known_filters = {f.filter for f in usable if f.imagetyp in (LIGHT, FLAT) and f.filter}
    agg: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for f in usable:
        for code, msg in folder_consistency(f, known_filters):
            agg[(code, str(Path(f.rel_path).parent), msg)].append(f.rel_path)
    for (code, folder, msg), files in sorted(agg.items()):
        anomalies.append(Anomaly("warning", code, f"{folder} : {len(files)} fichier(s), {msg}", files))

    if usable and all(f.offset is None for f in usable):
        anomalies.append(Anomaly("warning", "offset_missing",
                                 "OFFSET absent de tous les en-têtes → offset considéré constant"))
    elif any(f.offset is None for f in usable):
        anomalies.append(Anomaly("warning", "offset_partial",
                                 "OFFSET absent de certains en-têtes seulement → clés capteur distinctes"))

    lights = [f for f in usable if f.imagetyp == LIGHT]
    darks = [f for f in usable if f.imagetyp == DARK]
    flats = [f for f in usable if f.imagetyp == FLAT]
    darkflats = [f for f in usable if f.imagetyp == DARKFLAT]
    biases = [f for f in usable if f.imagetyp == BIAS]
    if biases:
        anomalies.append(Anomaly("info", "bias_unused",
                                 f"{len(biases)} bias trouvés : non utilisés (stratégie B, capteur CMOS)"))
    no_filter = [f.rel_path for f in lights + flats if not f.filter]
    if no_filter:
        anomalies.append(Anomaly("warning", "no_filter", "lights/flats sans mot-clé FILTER (filtre « ? »)",
                                 no_filter))
        for f in lights + flats:
            if not f.filter:
                f.filter = "?"

    sets = {
        DARK: build_sets(darks, DARK, gap_hours),
        FLAT: build_sets(flats, FLAT, gap_hours),
        DARKFLAT: build_sets(darkflats, DARKFLAT, gap_hours),
    }

    # Light groups: (session, filter, sensor key, exposure).
    groups: dict[tuple, list[Frame]] = defaultdict(list)
    for f in lights:
        groups[(f.session_date(offset), f.filter, f.sensor_key, _exp_bucket(f.exptime))].append(f)
    session_frames: dict[dt.date, list[Frame]] = defaultdict(list)
    for f in lights:
        session_frames[f.session_date(offset)].append(f)
    session_labels: dict[dt.date, str | None] = {}
    for sess, members in session_frames.items():
        labels = Counter(f.folder_label for f in members if f.folder_label)
        session_labels[sess] = labels.most_common(1)[0][0] if labels else None
        if len(labels) > 1:
            anomalies.append(Anomaly("warning", "session_labels",
                                     f"session {sess} : lights dans plusieurs dossiers SESSION "
                                     f"({', '.join(sorted(labels))})"))
    session_span = {s: (min(f.date_obs for f in fr), max(f.date_obs for f in fr))
                    for s, fr in session_frames.items()}

    light_groups: list[LightGroup] = []
    for (sess, filt, key, exp), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3])):
        g = LightGroup(sess, session_labels.get(sess), filt, key, exp, sorted(members, key=lambda f: f.date_obs))
        g.dark = match_darks(g, sets[DARK], cfg)
        g.flat = match_flats(g, session_span[sess], sets[FLAT], cfg)
        if g.flat.level != "C" and g.flat.sets:
            g.darkflat = match_darkflats(g.flat.sets[0], sets[DARKFLAT], sets[DARK], cfg)
            if g.darkflat.level == "C":
                g.flat = Match("C", [], g.flat.reasons + ["flats écartés faute de dark de flat"])
        light_groups.append(g)

    sensors = _sensor_infos(usable, bool(inv_cfg.get("read_pixels_for_units", True)), anomalies)
    if len(sensors) > 1:
        anomalies.append(Anomaly("info", "several_sensors",
                                 f"{len(sensors)} configurations capteur distinctes (les appariements "
                                 "exigent une clé identique)"))

    unused = [s for kind in (DARK, FLAT, DARKFLAT) for s in sets[kind]
              if not any(s in (m.sets if m else []) for g in light_groups
                         for m in (g.dark, g.flat, g.darkflat))]
    if unused:
        anomalies.append(Anomaly("info", "unused_sets",
                                 f"{len(unused)} série(s) de calibration non utilisée(s) : "
                                 + ", ".join(s.id for s in unused)))
    if disc.empty_dirs:
        anomalies.append(Anomaly("info", "empty_dirs", f"{len(disc.empty_dirs)} dossier(s) vide(s)",
                                 disc.empty_dirs))
    if disc.ignored_dirs:
        anomalies.append(Anomaly("info", "ignored_dirs", f"{len(disc.ignored_dirs)} dossier(s) ignoré(s)",
                                 disc.ignored_dirs))

    return Inventory(root, cfg, usable, light_groups, sets, sensors, anomalies, disc.ignored_dirs,
                     disc.empty_dirs, dict(disc.other_extensions), dt.datetime.now().replace(microsecond=0))


# --------------------------------------------------------------------------- darks to redo


def darks_todo(inv: Inventory) -> list[DarkTodo]:
    """Combinations for which level-A calibration darks are missing."""
    todo: dict[tuple, DarkTodo] = {}
    for g in inv.light_groups:
        if g.dark and g.dark.level != "A" and not g.dark.borrowed_night:
            k = ("dark", g.sensor_key, g.exptime, g.mean_set_temp)
            item = todo.get(k)
            if item is None:
                item = todo[k] = DarkTodo("dark", g.sensor_key, g.exptime, g.mean_set_temp, g.mean_temp,
                                          [], [], g.dark.level, "; ".join(g.dark.reasons))
            item.filters = sorted(set(item.filters) | {g.filter})
            item.sessions = sorted(set(item.sessions) | {inv.session_label(g.session)})
        if g.darkflat and g.darkflat.level != "A" and g.flat and g.flat.sets:
            fs = g.flat.sets[0]
            set_temps = [f.set_temp for f in fs.frames if f.set_temp is not None]
            st = statistics.fmean(set_temps) if set_temps else None
            k = ("darkflat", fs.sensor_key, fs.exptime, st)
            item = todo.get(k)
            if item is None:
                item = todo[k] = DarkTodo("darkflat", fs.sensor_key, fs.exptime, st, fs.mean_temp,
                                          [], [], g.darkflat.level, "; ".join(g.darkflat.reasons))
            item.filters = sorted(set(item.filters) | {g.filter})
            item.sessions = sorted(set(item.sessions) | {inv.session_label(g.session)})
    return sorted(todo.values(), key=lambda t: (t.kind, t.exptime))
