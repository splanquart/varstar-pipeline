"""Markdown rendering (French) of the quality checks: qa.md and the QA sections of other reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ...domain.qa.model import QAItem, severity_counts

SEVERITY_FR = {"error": "erreur", "warning": "avertissement", "info": "info"}


def item_markdown(i: QAItem) -> list[str]:
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


def summary_line(items: Iterable[QAItem]) -> str:
    n = severity_counts(items)
    return f"{n['error']} erreur(s), {n['warning']} avertissement(s), {n['info']} info(s)"


def items_markdown(items: list[QAItem], title: str = "## Contrôles qualité") -> list[str]:
    L = [title, "", f"_{summary_line(items)} — détail et méthode : `qa/qa.md`, docs/QA_CHECKS.md._", ""]
    if not items:
        return L + ["Aucun problème détecté.", ""]
    for i in items:
        L += item_markdown(i)
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
            L += item_markdown(i)
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


class QAReportAdapter:
    """``QAReport``: qa.md."""

    def markdown(self, report: dict[str, Any]) -> str:
        return qa_markdown(report)
