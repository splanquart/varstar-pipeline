"""``QAItem`` and the helpers shared by the quality rules."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from ..model import Inventory, LightGroup
from ..settings import Settings

SEVERITIES = ("error", "warning", "info")
STAGES = ("header", "raw", "calibrated")

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
    "P9": "Darks à une autre température",
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


def sort_items(items: Iterable[QAItem]) -> list[QAItem]:
    """Errors first, then warnings, then infos; by check within a severity (stable)."""
    return sorted(items, key=lambda i: (SEVERITIES.index(i.severity), i.check))


def report_items(report: Mapping[str, Any], stages: Iterable[str] = STAGES) -> list[QAItem]:
    """Items of the given stages of a qa.json report, sorted."""
    return sort_items(QAItem(**d) for s in stages for d in report.get(s, []))


def severity_counts(items: Iterable[QAItem]) -> dict[str, int]:
    items = list(items)
    return {s: sum(1 for i in items if i.severity == s) for s in SEVERITIES}


def night_label(inv: Inventory, night: dt.date) -> str:
    labels = {g.label for g in inv.light_groups if g.session == night and g.label}
    return f"{night:%d/%m}" + (f" ({', '.join(sorted(labels))})" if labels else "")


def group_label(g: LightGroup) -> str:
    return f"{g.label or g.session.isoformat()} / {g.filter}"


def offset_hours(cfg: Settings) -> float:
    return cfg.inventory.session_day_offset_hours
