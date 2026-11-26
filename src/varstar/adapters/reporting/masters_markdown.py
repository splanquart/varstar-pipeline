"""Markdown rendering (French) of masters/masters.md."""

from __future__ import annotations

from typing import Any

from ...domain.masks import BIT_NAMES, BIT_NAMES_FR
from ...domain.qa.model import QAItem
from .qa_markdown import items_markdown


def _f(x: Any, fmt: str = ".1f") -> str:
    return "—" if x is None else format(x, fmt)


def masters_markdown(plan: dict[str, Any], qa_items: list[QAItem] | None, masks: dict[str, dict[str, Any]]) -> str:
    """``qa_items``: header/raw findings (None: no QA report yet); ``masks``: mask path → counts/info."""
    L = ["# Masters et masques", "", f"- Bibliothèque : `{plan['library']}`",
         f"- Siril {plan['siril_version']} — généré le {plan['created']}", ""]
    if qa_items is not None:
        L += items_markdown(qa_items)
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
        if g.get("bpm") in masks:
            c = masks[g["bpm"]]["counts"]
            bpm_info = ", ".join(f"{BIT_NAMES_FR[b]} {c[n]}" for b, n in BIT_NAMES.items() if c.get(n))
        if g.get("ampglow") in masks:
            i = masks[g["ampglow"]]["info"]
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


class MastersReportAdapter:
    """``MastersReport``: masters.md."""

    def markdown(self, plan: dict[str, Any], qa_items: list[QAItem] | None,
                 masks: dict[str, dict[str, Any]]) -> str:
        return masters_markdown(plan, qa_items, masks)
