"""Post-calibration checks (C*): from the statistics of the calibrated frames of each group.

``per`` lists, for every calibrated frame of a group, ``{"file", "raw", **calibrated_frame_stats}``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from .model import QAItem


def calibrated_group_checks(key: str, group: Mapping[str, Any], per: list[dict[str, Any]],
                            cfg: dict[str, Any]) -> tuple[list[QAItem], dict[str, Any] | None]:
    """C3 findings and the table row of one calibrated group (``group``: its entry of the masters plan)."""
    bg_min = float(cfg["qa"]["calibrated_background_min_adu"])
    items: list[QAItem] = []
    if not per:
        return items, None
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
    lbl = row["label"]
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
    for i in items:
        i.data["group"] = key
    return items, row


def ampglow_residual_items(rows: list[dict[str, Any]], limit: float = 3.0) -> list[QAItem]:
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

