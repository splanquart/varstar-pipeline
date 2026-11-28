"""astrometry/astrometry.md (French)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .qa_markdown import items_markdown


def _f(x: Any, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


def astrometry_markdown(summary: dict[str, Any], qa_items: list[Any]) -> str:
    L = ["# Astrométrie", "", f"- Solveur : {summary['solver']}", f"- Catalogue : {summary.get('catalog') or '—'}",
         f"- Images résolues : {summary['n_solved']} / {summary['n_images']}", "",
         "Les images calibrées ne sont pas modifiées : la WCS de chaque image (mots-clés FITS, distorsion SIP "
         "comprise) est enregistrée dans `astrometry/<groupe>.json`.", ""]
    L += items_markdown(qa_items, "## Contrôles (A*)")
    L += ["## Groupes", "", "| Groupe | Images | Résolues | Résidus médians ″ | Échelle ″/px | Orientation ° | "
          "Segments |", "|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        L.append(f"| {r['label']} | {r['n']} | {r['n_solved']} | {_f(r['rms_median_arcsec'], '.2f')} | "
                 f"{_f(r['scale_median_arcsec'], '.4f')} | {_f(r['orientation_median_deg'], '.1f')} | "
                 f"{', '.join(str(s) for s in r['segments']) or '—'} |")
    L.append("")
    for key, res in summary["groups"].items():
        L += [f"### {key}", "", "| Image | Résolue | Étoiles | Appariées | Résidus ″ | Orientation ° | Segment | "
              "Raison |", "|---|---|---|---|---|---|---|---|"]
        for raw, row in sorted(res["images"].items(), key=lambda kv: (kv[1].get("time") or "", kv[0])):
            L.append(f"| `{Path(raw).name}` | {'oui' if row['ok'] else '**non**'} | {row['n_stars']} | "
                     f"{row['n_matched']} | {_f(row.get('rms_arcsec'), '.2f')} | "
                     f"{_f(row.get('orientation_deg'), '.2f')} | {row.get('segment') or '—'} | "
                     f"{row.get('reason') or ''} |")
        L.append("")
    if summary.get("errors"):
        L += ["## Groupes non traités", ""] + [f"- {k} : {v}" for k, v in summary["errors"].items()] + [""]
    return "\n".join(L) + "\n"


class AstrometryReportAdapter:
    """``AstrometryReport``."""

    def markdown(self, summary: dict[str, Any], qa_items: list[Any]) -> str:
        return astrometry_markdown(summary, qa_items)
