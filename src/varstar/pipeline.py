"""Pipeline driver: inventory -> masters -> calibrate (steps after M2 are not implemented yet)."""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .calibrate import calibrate_group
from .fitsinfo import Frame
from .imageio import NORM_SCALE, XisfError, read_xisf, write_mask
from .inventory import Inventory, LightGroup, build_inventory
from .masks import BIT_NAMES, BIT_NAMES_FR, build_amp_glow, build_bpm
from .masters import BuildContext, Library, MasterRecord, build_dark_master, build_flat_master
from .provenance import config_fingerprint, update_provenance
from . import qa
from .siril import Siril
from .steps import STEPS, run_inventory, setup_logging

log = logging.getLogger("varstar")

IMPLEMENTED = ["inventory", "check", "masters", "calibrate"]


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- selection


def group_key(g: LightGroup) -> str:
    return f"{g.session.isoformat()}_{g.filter}_{g.exptime:g}s"


def select_groups(inv: Inventory, sessions: list[str] | None, filters: list[str] | None) -> list[LightGroup]:
    out = []
    for g in inv.light_groups:
        if sessions and not ({g.session.isoformat(), g.label or ""} & set(sessions)):
            continue
        if filters and g.filter not in filters:
            continue
        out.append(g)
    if sessions:
        known = {g.session.isoformat() for g in inv.light_groups} | {g.label for g in inv.light_groups if g.label}
        unknown = set(sessions) - known
        if unknown:
            raise PipelineError(f"Session(s) inconnue(s) : {', '.join(sorted(unknown))}")
    return out


def _frames(sets) -> list[Frame]:
    return sorted((f for s in sets for f in s.frames), key=lambda f: f.date_obs)


# --------------------------------------------------------------------------- masters step


def _offset_level(inv: Inventory, g: LightGroup, ctx: BuildContext) -> tuple[float, str]:
    """Uniform offset estimate for dark scaling: median of the shortest dark available."""
    pool = [s for kind in ("DARKFLAT", "DARK") for s in inv.calibration_sets[kind]
            if s.sensor_key == g.sensor_key]
    if not pool:
        raise PipelineError("aucun dark court pour estimer l'offset")
    shortest = min(s.exptime for s in pool)
    sets = [s for s in pool if s.exptime == shortest]
    kind = "darkflat" if sets[0].kind == "DARKFLAT" else "dark"
    rec = build_dark_master(_frames(sets), kind, ctx, inv.root)
    return float(rec.meta["median_adu"]), rec.id


def _crosscheck(rec: MasterRecord, pi_dir: Path | None, folder_labels: list[str]) -> dict[str, Any] | None:
    if pi_dir is None or not pi_dir.exists():
        return None
    meta = rec.meta
    if rec.kind in ("dark", "darkflat"):
        pattern = f"masterDark*EXPOSURE-{meta['exptime']:.2f}s*.xisf"
    else:
        pattern = None
        for lab in folder_labels:
            cands = sorted(pi_dir.glob(f"masterFlat*FILTER-{meta['filter']}_*SESSION-{lab}.xisf"))
            if cands:
                pattern = cands[0].name
                break
        if pattern is None:
            return None
    cands = sorted(pi_dir.glob(pattern))
    if not cands:
        return None
    pi_path = cands[0]
    try:
        theirs = read_xisf(pi_path).astype(np.float32) * np.float32(NORM_SCALE)
    except (XisfError, KeyError, ValueError) as exc:
        return {"pixinsight": pi_path.name, "error": str(exc)}
    ours = rec.load_adu()
    if theirs.shape != ours.shape:
        return {"pixinsight": pi_path.name, "error": f"dimensions différentes {theirs.shape} / {ours.shape}"}
    a = ours[::2, ::2]
    best = None
    for flipped in (False, True):
        b = (theirs[::-1] if flipped else theirs)[::2, ::2]
        if rec.kind == "flat":
            r = (a / np.median(a)) / (b / np.median(b)) - 1.0
            stat = {"robust_sigma_ratio_pct": 100 * _rs(r), "median_ratio_pct": 100 * float(np.median(r))}
            score = stat["robust_sigma_ratio_pct"]
        else:
            d = a - b
            stat = {"median_ours_adu": float(np.median(a)), "median_pi_adu": float(np.median(b)),
                    "median_diff_adu": float(np.median(d)), "robust_sigma_diff_adu": _rs(d)}
            score = stat["robust_sigma_diff_adu"]
        if best is None or score < best[0]:
            best = (score, flipped, stat)
    return {"pixinsight": pi_path.name, "flipped": best[1], **best[2]}


def _rs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _dark_consistency(rec: MasterRecord) -> list[dict[str, Any]]:
    """Q2: compare the dark sub-groups (by folder, i.e. INTENSITY_dark / INTENSITY_light)."""
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for s in rec.meta.get("frame_stats", []):
        by_folder[str(Path(s["file"]).parent)].append(s)
    rows = []
    for folder, st in sorted(by_folder.items()):
        corners = {k: float(np.mean([x["corners"][k] for x in st])) for k in st[0]["corners"]}
        rows.append({"folder": folder, "n": len(st), "median_adu": float(np.mean([x["median"] for x in st])),
                     "robust_sigma_adu": float(np.mean([x["robust_sigma"] for x in st])),
                     "corner_excess_adu": corners,
                     "mean_ccd_temp": float(np.mean([x["ccd_temp"] for x in st if x["ccd_temp"] is not None]))})
    return rows


def run_masters(inv: Inventory, groups: list[LightGroup], cfg: dict[str, Any], workdir: Path,
                siril: Siril) -> dict[str, Any]:
    setup_logging(workdir, "masters", cfg["runtime"].get("log_level", "INFO"))
    library = Library(Path(cfg["paths"]["library"]).expanduser())
    ctx = BuildContext(cfg, siril, library, workdir / "siril")
    pi_cfg = cfg["paths"].get("pixinsight_masters", "auto")
    pi_dir = (inv.root / "output" / "master") if pi_cfg == "auto" else (Path(pi_cfg) if pi_cfg else None)
    sensor = inv.sensors[0] if inv.sensors else None

    masters: dict[str, MasterRecord] = {}
    flat_labels: dict[str, list[str]] = {}
    plan: dict[str, Any] = {"created": dt.datetime.now().isoformat(timespec="seconds"),
                            "library": str(library.root), "siril_version": siril.version(), "groups": {}}
    warnings: list[str] = []
    saturation = float(sensor.saturation_adu) if sensor and sensor.saturation_adu else 65532.0
    stats_cache = qa.StatsCache(workdir, cfg, saturation)

    for g in groups:
        key = group_key(g)
        print(f"  masters pour {g.label or g.session} / {g.filter} …", flush=True)
        entry: dict[str, Any] = {"key": key, "session": g.session.isoformat(), "label": g.label,
                                 "filter": g.filter, "exptime": g.exptime, "level": g.level,
                                 "reasons": g.reasons, "files": [str(f.path) for f in g.frames],
                                 "dark": None, "flat": None, "darkflat": None, "bpm": None, "ampglow": None}
        dark_rec = None
        dark_sets = list(g.dark.sets) if g.dark else []
        if dark_sets:
            pool = {s.id: s for s in qa.dark_candidates(inv, g, cfg) + dark_sets}
            stats = stats_cache.get(_frames(pool.values()), progress="niveau des darks")
            dark_sets, notes = qa.effective_dark_sets(inv, g, stats, cfg)
            for note in notes:
                entry["reasons"] = entry["reasons"] + [note]
                warnings.append(f"{key} : {note}")
        if dark_sets:
            dark_rec = build_dark_master(_frames(dark_sets), "dark", ctx, inv.root)
            masters[dark_rec.id] = dark_rec
            mode = "scaled" if any(abs(s.exptime - g.exptime) > cfg["matching"]["exptime_tolerance_s"]
                                   for s in dark_sets) else "direct"
            entry["dark"] = {"id": dark_rec.id, "path": str(dark_rec.path), "mode": mode,
                             "n": dark_rec.meta["n_frames"], "exptime": dark_rec.meta["exptime"]}
            if mode == "scaled":
                off, off_id = _offset_level(inv, g, ctx)
                entry["dark"].update(offset_adu=off, offset_master=off_id)
                warnings.append(f"{key} : dark mis à l'échelle depuis {dark_rec.meta['exptime']:g} s "
                                f"(offset estimé {off:.1f} ADU) — niveau B")
        flat_rec = None
        if g.flat and g.flat.level != "C" and g.flat.sets and g.darkflat and g.darkflat.sets:
            df_kind = "darkflat" if g.darkflat.sets[0].kind == "DARKFLAT" else "dark"
            df_rec = build_dark_master(_frames(g.darkflat.sets), df_kind, ctx, inv.root)
            masters[df_rec.id] = df_rec
            flat_rec = build_flat_master(_frames(g.flat.sets), df_rec, ctx, inv.root)
            masters[flat_rec.id] = flat_rec
            flat_labels[flat_rec.id] = g.flat.sets[0].folder_labels
            entry["darkflat"] = {"id": df_rec.id, "path": str(df_rec.path), "exptime": df_rec.meta["exptime"]}
            entry["flat"] = {"id": flat_rec.id, "path": str(flat_rec.path), "n": flat_rec.meta["n_frames"]}
            if flat_rec.meta.get("level_warnings"):
                warnings.append(f"{flat_rec.meta['label']} : {len(flat_rec.meta['level_warnings'])} flat(s) "
                                "hors de la plage 15–85 % de la dynamique")

        # Masks (per dark/flat pair), stored in the workdir.
        if dark_rec is not None or flat_rec is not None:
            mhash = config_fingerprint(cfg, ["masks"])[:8]
            mid = f"{dark_rec.id if dark_rec else 'nodark'}_{flat_rec.id if flat_rec else 'noflat'}_{mhash}"
            bpm_path = workdir / "masks" / f"bpm_{mid}.fits"
            dark_adu = dark_rec.load_adu() if dark_rec is not None else None
            if not bpm_path.exists():
                unstable = (fits.getdata(dark_rec.unstable_path) if dark_rec is not None
                            and dark_rec.unstable_path.exists() else None)
                res = build_bpm(dark_adu, dark_rec.meta.get("sigma_single_frame_adu") if dark_rec else None,
                                unstable, flat_rec.load_adu() if flat_rec is not None else None, cfg,
                                saturation_adu=(sensor.saturation_adu if sensor and sensor.saturation_adu
                                                else 65532))
                hdr = fits.Header()
                hdr["VSDARK"] = dark_rec.id if dark_rec else "NONE"
                hdr["VSFLAT"] = flat_rec.id if flat_rec else "NONE"
                for bit, name in BIT_NAMES.items():
                    hdr[f"VSBIT{bit}"] = name
                write_mask(bpm_path, res.mask, hdr)
                (bpm_path.with_suffix(".json")).write_text(json.dumps(
                    {"counts": res.counts, "info": res.info}, indent=2), encoding="utf-8")
            entry["bpm"] = str(bpm_path)
            if dark_rec is not None and cfg["masks"].get("amp_glow_detect", True):
                ag_path = workdir / "masks" / f"ampglow_{dark_rec.id}_{mhash}.fits"
                if not ag_path.exists():
                    res = build_amp_glow(dark_adu, cfg)
                    write_mask(ag_path, res.mask)
                    ag_path.with_suffix(".json").write_text(json.dumps(
                        {"counts": res.counts, "info": res.info}, indent=2), encoding="utf-8")
                entry["ampglow"] = str(ag_path)
        plan["groups"][key] = entry

    # Cross-validation with PixInsight masters + Q2 dark consistency.
    plan["masters"] = {}
    for rec in masters.values():
        info = {"kind": rec.kind, "label": rec.meta["label"], "dir": str(rec.dir), "n": rec.meta["n_frames"],
                "rejection": rec.meta["rejection"]["name"], "median_adu": rec.meta["median_adu"],
                "exptime": rec.meta["exptime"], "filter": rec.meta.get("filter"),
                "sigma_single_frame_adu": rec.meta.get("sigma_single_frame_adu"),
                "n_unstable": rec.meta.get("n_unstable"),
                "level_min_frac": rec.meta.get("level_min_frac"), "level_max_frac": rec.meta.get("level_max_frac")}
        cc = _crosscheck(rec, pi_dir, flat_labels.get(rec.id, []))
        if cc:
            info["pixinsight"] = cc
        if rec.kind == "dark":
            info["consistency"] = _dark_consistency(rec)
        plan["masters"][rec.id] = info
    plan["warnings"] = warnings
    (workdir / "masters").mkdir(parents=True, exist_ok=True)
    (workdir / "masters" / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    (workdir / "masters" / "masters.md").write_text(masters_markdown(plan, workdir), encoding="utf-8")
    update_provenance(workdir, "masters", cfg, {
        "siril_version": siril.version(), "library": str(library.root),
        "masters": {mid: {"kind": m.kind, "dir": str(m.dir), "sources": m.meta["sources"],
                          "rejection": m.meta["rejection"], "n": m.meta["n_frames"]}
                    for mid, m in masters.items()},
    })
    for w in warnings:
        log.warning(w)
    return plan


# --------------------------------------------------------------------------- report (markdown)


def _f(x: Any, fmt: str = ".1f") -> str:
    return "—" if x is None else format(x, fmt)


def masters_markdown(plan: dict[str, Any], workdir: Path) -> str:
    L = ["# Masters et masques", "", f"- Bibliothèque : `{plan['library']}`",
         f"- Siril {plan['siril_version']} — généré le {plan['created']}", ""]
    report = qa.load_report(workdir)
    if report:
        L += qa.items_markdown(qa.all_items(report, ("header", "raw")))
    L += ["## Masters", "", "| Type | Master | N | Rejet | Médiane ADU | σ 1 pose ADU | Pixels instables | "
          "Niveau flats (% dyn.) |", "|---|---|---|---|---|---|---|---|"]
    kinds = {"dark": "dark", "darkflat": "dark de flat", "flat": "flat"}
    for mid, m in sorted(plan["masters"].items(), key=lambda kv: (kv[1]["kind"], kv[1]["label"])):
        lvl = (f"{100 * m['level_min_frac']:.0f}–{100 * m['level_max_frac']:.0f}"
               if m.get("level_min_frac") is not None else "—")
        L.append(f"| {kinds[m['kind']]} | `{m['label']}` ({mid[:8]}) | {m['n']} | {m['rejection']} | "
                 f"{_f(m['median_adu'])} | {_f(m.get('sigma_single_frame_adu'), '.2f')} | "
                 f"{m.get('n_unstable') if m.get('n_unstable') is not None else '—'} | {lvl} |")
    L += ["", "## Groupes de lights", "", "| Groupe | Niveau | Dark | Flat | Dark de flat | Masque BPM | "
          "Amp glow |", "|---|---|---|---|---|---|---|"]
    for key, g in plan["groups"].items():
        bpm_info, ag_info = "—", "—"
        if g.get("bpm"):
            bj = Path(g["bpm"]).with_suffix(".json")
            if bj.exists():
                c = json.loads(bj.read_text())["counts"]
                bpm_info = ", ".join(f"{BIT_NAMES_FR[b]} {c[n]}" for b, n in BIT_NAMES.items() if c.get(n))
        if g.get("ampglow"):
            aj = Path(g["ampglow"]).with_suffix(".json")
            if aj.exists():
                i = json.loads(aj.read_text())["info"]
                ag_info = f"{100 * i['fraction']:.1f} % du champ (excès max {i['max_excess_adu']:.0f} ADU)"
        dark = (f"{g['dark']['n']} × {g['dark']['exptime']:g} s" + (" (mis à l'échelle)"
                if g["dark"]["mode"] == "scaled" else "")) if g.get("dark") else "aucun"
        flat = f"{g['flat']['n']} flats" if g.get("flat") else "aucun"
        dflat = f"{g['darkflat']['exptime']:g} s" if g.get("darkflat") else "—"
        L.append(f"| {g['label'] or g['session']} / {g['filter']} ({g['exptime']:g} s) | **{g['level']}** | "
                 f"{dark} | {flat} | {dflat} | {bpm_info} | {ag_info} |")
    L += ["", "Masque BPM : *tièdes* = excès de dark > k·σ d'une pose (bien corrigés par le dark, "
          "information seulement) ; *très chauds*, *instables*, *froids*, *morts*, *saturés* = pixels "
          "dont la correction est douteuse. Les pixels ne sont jamais modifiés : les mesures touchées "
          "seront marquées (jalon M4).", ""]

    L += ["## Validation croisée avec les masters PixInsight", ""]
    rows = [(mid, m) for mid, m in plan["masters"].items() if m.get("pixinsight")]
    if not rows:
        L.append("Aucun master PixInsight correspondant trouvé.")
    else:
        L += ["| Master | PixInsight | Résultat |", "|---|---|---|"]
        for mid, m in rows:
            cc = m["pixinsight"]
            if "error" in cc:
                res = f"comparaison impossible : {cc['error']}"
            elif m["kind"] == "flat":
                res = (f"écart relatif : médiane {cc['median_ratio_pct']:+.3f} %, dispersion "
                       f"{cc['robust_sigma_ratio_pct']:.3f} %")
            else:
                res = (f"médianes {cc['median_ours_adu']:.1f} / {cc['median_pi_adu']:.1f} ADU, différence "
                       f"{cc['median_diff_adu']:+.2f} ADU, dispersion {cc['robust_sigma_diff_adu']:.2f} ADU")
            if cc.get("flipped"):
                res += " — **image PixInsight retournée verticalement**"
            L.append(f"| `{m['label']}` | `{cc['pixinsight']}` | {res} |")
    L += ["", "## Cohérence des darks par dossier (question Q2)", ""]
    for mid, m in plan["masters"].items():
        if m.get("consistency") and len(m["consistency"]) > 1:
            L += [f"### `{m['label']}`", "", "| Dossier | N | Niveau médian ADU | σ ADU | T moy. °C | "
                  "Excès coins (HG/HD/BG/BD) ADU |", "|---|---|---|---|---|---|"]
            for r in m["consistency"]:
                c = r["corner_excess_adu"]
                L.append(f"| {r['folder']} | {r['n']} | {r['median_adu']:.1f} | {r['robust_sigma_adu']:.2f} | "
                         f"{r['mean_ccd_temp']:.1f} | {c['haut_gauche']:.1f} / {c['haut_droit']:.1f} / "
                         f"{c['bas_gauche']:.1f} / {c['bas_droit']:.1f} |")
            meds = [r["median_adu"] for r in m["consistency"]]
            spread = max(meds) - min(meds)
            verdict = ("différence significative — à examiner" if spread > 2.0
                       else "pas de différence significative")
            L += ["", f"Écart des niveaux médians : {spread:.1f} ADU → {verdict}.", ""]
    if plan.get("warnings"):
        L += ["## Avertissements", ""] + [f"- {w}" for w in plan["warnings"]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- calibrate step


def run_calibrate(plan: dict[str, Any], inv: Inventory, keys: list[str], cfg: dict[str, Any], workdir: Path,
                  siril: Siril, force: bool = False) -> dict[str, Any]:
    setup_logging(workdir, "calibrate", cfg["runtime"].get("log_level", "INFO"))
    s = inv.sensors[0] if inv.sensors else None
    sensor = {"adu_scale": s.adu_scale if s else None, "effective_gain": s.effective_gain if s else None}
    results, errors = {}, {}
    for key in keys:
        group = plan["groups"].get(key)
        if group is None:
            errors[key] = "groupe absent du plan des masters (relancer l'étape masters)"
            continue
        print(f"  calibration {group['label'] or group['session']} / {group['filter']} "
              f"({len(group['files'])} images) …", flush=True)
        try:
            results[key] = calibrate_group(group, workdir, siril, sensor, force=force)
        except Exception as exc:  # one failing group must not stop the others
            log.error("Calibration %s en échec : %s", key, exc)
            errors[key] = str(exc)
    if results:
        items, rows = qa.calibrated_checks(plan, results, cfg)
        report = qa.load_report(workdir)
        old = {r["group"]: r for r in report.get("tables", {}).get("calibrated", [])}
        old.update({r["group"]: r for r in rows})
        keep = [d for d in report.get("calibrated", []) if d.get("data", {}).get("group") not in results]
        report["calibrated"] = keep + [i.to_dict() for i in items]
        report.setdefault("tables", {})["calibrated"] = list(old.values())
        qa.save_report(workdir, report)
        for i in items:
            getattr(log, "error" if i.severity == "error" else "warning")(f"[{i.check}] {i.message}")
    update_provenance(workdir, "calibrate", cfg, {
        "images": {raw: out for r in results.values() for raw, out in r["outputs"].items()},
        "groups": {k: {"level": plan["groups"][k]["level"], "dark": plan["groups"][k]["dark"],
                       "flat": plan["groups"][k]["flat"], "darkflat": plan["groups"][k]["darkflat"],
                       "bpm": plan["groups"][k]["bpm"], "ampglow": plan["groups"][k]["ampglow"]}
                   for k in results},
        "errors": errors,
    })
    return {"results": results, "errors": errors, "qa": qa.load_report(workdir).get("calibrated", [])}


# --------------------------------------------------------------------------- driver


def run_pipeline(source: Path, cfg: dict[str, Any], workdir: Path, from_step: str = "inventory",
                 sessions: list[str] | None = None, filters: list[str] | None = None,
                 force: bool = False) -> dict[str, Any]:
    if from_step not in STEPS:
        raise PipelineError(f"Étape inconnue : {from_step} (étapes : {', '.join(STEPS)})")
    start = STEPS.index(from_step)
    summary: dict[str, Any] = {"steps": []}

    if start <= STEPS.index("inventory"):
        run_inventory(source, cfg, workdir, force=force)
        summary["steps"].append("inventory")
    inv = build_inventory(source, cfg, exclude=[workdir])
    groups = select_groups(inv, sessions, filters)
    if not groups:
        raise PipelineError("Aucun groupe de lights ne correspond à la sélection")
    summary["groups"] = [group_key(g) for g in groups]
    if start <= STEPS.index("check"):
        summary["qa"] = qa.run_checks(inv, cfg, workdir)
        summary["steps"].append("check")
    siril = Siril.from_config(cfg, workdir)

    plan_path = workdir / "masters" / "plan.json"
    if start <= STEPS.index("masters"):
        plan = run_masters(inv, groups, cfg, workdir, siril)
        summary["steps"].append("masters")
    else:
        if not plan_path.exists():
            raise PipelineError("plan des masters absent : relancer depuis l'étape « masters »")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary["plan"] = plan

    if start <= STEPS.index("calibrate"):
        summary["calibrate"] = run_calibrate(plan, inv, summary["groups"], cfg, workdir, siril, force=force)
        summary["steps"].append("calibrate")

    remaining = [s for s in STEPS[max(start, STEPS.index("calibrate") + 1):] if s not in IMPLEMENTED]
    summary["not_implemented"] = remaining
    return summary
