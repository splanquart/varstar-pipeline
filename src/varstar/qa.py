"""Legacy entry points of the quality checks (re-exports until the composition root, step 6).

The rules are in ``varstar.domain.qa``, the use case in ``varstar.application.check``, the pixel
measurements in ``varstar.adapters.pixel_stats`` and the Markdown in
``varstar.adapters.reporting.qa_markdown``. The functions below wire the file adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .adapters.pixel_stats import PixelStatsCache, binned_frame, quick_stats  # noqa: F401
from .adapters.reporting.qa_markdown import (SEVERITY_FR, QAReportAdapter, items_markdown,  # noqa: F401
                                             qa_markdown, summary_line)
from .adapters.workspace import Workspace
from .application import check as _check
from .domain.model import Frame, Inventory
from .domain.qa.darks import (classify_excess, consistent_dark_sets, dark_candidates,  # noqa: F401
                              effective_dark_sets, level_clusters, set_summary, usual_level_dark_sets)
from .domain.qa.headers import header_checks  # noqa: F401
from .domain.qa.model import SEVERITIES, STAGES, QAItem, report_items  # noqa: F401
from .domain.qa.pixels import flat_ratio_metrics  # noqa: F401


def _say(message: str) -> None:
    print(message, flush=True)


class StatsCache(PixelStatsCache):
    """Former constructor: ``StatsCache(workdir, cfg, saturation).get(frames)``."""

    def __init__(self, workdir: Path, cfg: dict[str, Any], saturation: float):
        super().__init__(workdir, step=int(cfg["qa"].get("pixel_step", 4)),
                         processes=int(cfg.get("runtime", {}).get("processes", 1)))
        self.saturation = saturation

    def get(self, frames: list[Frame], progress: str | None = None) -> dict[str, dict[str, Any]]:
        return self.frame_stats(frames, self.saturation, progress)


def raw_checks(inv: Inventory, cfg: dict[str, Any], workdir: Path) -> tuple[list[QAItem], dict[str, Any]]:
    return _check.raw_checks(inv, cfg, store=Workspace(workdir), pixels=PixelStatsCache.from_config(workdir, cfg),
                             progress=_say)


def calibrated_checks(plan: dict[str, Any], results: dict[str, Any], cfg: dict[str, Any],
                      step: int = 4) -> tuple[list[QAItem], list[dict[str, Any]]]:
    # Only images are read here: the frame statistics cache (never loaded) does not matter.
    return _check.calibrated_checks(plan, results, cfg, pixels=PixelStatsCache(Path(".")), step=step)


def load_report(workdir: Path) -> dict[str, Any]:
    return _check.load_report(Workspace(workdir))


def save_report(workdir: Path, report: dict[str, Any]) -> None:
    _check.save_report(Workspace(workdir), report, QAReportAdapter())


def record_calibrated(workdir: Path, items: list[QAItem], rows: list[dict[str, Any]],
                      groups: Iterable[str]) -> dict[str, Any]:
    return _check.record_calibrated_checks(Workspace(workdir), QAReportAdapter(), items, rows, groups)


def run_checks(inv: Inventory, cfg: dict[str, Any], workdir: Path, raw: bool = True) -> dict[str, Any]:
    return _check.run_checks(inv, cfg, store=Workspace(workdir), pixels=PixelStatsCache.from_config(workdir, cfg),
                             report=QAReportAdapter(), raw=raw, progress=_say)


def all_items(report: dict[str, Any], stages: Iterable[str] = STAGES) -> list[QAItem]:
    return report_items(report, stages)
