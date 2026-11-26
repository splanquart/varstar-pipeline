"""Composition root: instantiates the adapters and gives them to the use cases.

Nothing else in the code base builds an adapter (checked by tests/architecture/test_layers.py,
legacy modules excepted until step 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..adapters.console import ConsoleReporter, StepLog
from ..adapters.discovery import FilesystemFrameSource
from ..adapters.fits_io import FitsPixelReader
from ..adapters.library import DiskMasterRepository
from ..adapters.pixel_stats import PixelStatsCache
from ..adapters.pixinsight import PixInsightMasters
from ..adapters.provenance import ProvenanceStore
from ..adapters.reporting.inventory_report import InventoryReportAdapter
from ..adapters.reporting.masters_markdown import MastersReportAdapter
from ..adapters.reporting.qa_markdown import QAReportAdapter
from ..adapters.siril import Siril, SirilError
from ..adapters.siril_calibrator import SirilCalibrator
from ..adapters.siril_masters import SirilMasterBuilder
from ..adapters.workspace import Workspace
from ..application import check as check_use_case
from ..application import inventory as inventory_use_case
from ..application import pipeline as pipeline_use_case
from ..application.inventory import InventoryResult
from ..application.pipeline import PipelineServices
from ..domain.calibration import PipelineError
from ..domain.model import Inventory

USER_ERRORS = (PipelineError, SirilError)
"""Errors shown to the user as « ERREUR : … » (no traceback)."""


def _log_level(cfg: dict[str, Any]) -> str:
    return cfg.get("runtime", {}).get("log_level", "INFO")


def step_log(workdir: Path, cfg: dict[str, Any]) -> StepLog:
    return StepLog(workdir, _log_level(cfg))


def load_inventory(source: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> Inventory:
    """Inventory of a source folder (headers + ADU scale), nothing written."""
    return inventory_use_case.load_inventory(FilesystemFrameSource(), FitsPixelReader(), source, cfg, exclude)


def run_inventory(source: Path, cfg: dict[str, Any], workdir: Path, force: bool = False) -> InventoryResult:
    workdir.mkdir(parents=True, exist_ok=True)
    step_log(workdir, cfg).start("inventory")
    return inventory_use_case.run_inventory(
        source.resolve(), cfg, store=Workspace(workdir), provenance=ProvenanceStore(workdir),
        source=FilesystemFrameSource(), pixels=FitsPixelReader(), report=InventoryReportAdapter(), force=force)


def run_checks(source: Path, cfg: dict[str, Any], workdir: Path, raw: bool = True) -> dict[str, Any]:
    """« varstar check »: header (+ raw pixel) checks, qa/qa.json and qa/qa.md."""
    workdir.mkdir(parents=True, exist_ok=True)
    step_log(workdir, cfg).start("check")
    inv = load_inventory(source, cfg, exclude=[workdir])
    reporter = ConsoleReporter()
    return check_use_case.run_checks(inv, cfg, store=Workspace(workdir),
                                     pixels=PixelStatsCache.from_config(workdir, cfg, reporter.progress),
                                     report=QAReportAdapter(), raw=raw, progress=reporter.progress)


def pipeline_services(source: Path, cfg: dict[str, Any], workdir: Path) -> PipelineServices:
    reporter = ConsoleReporter()

    def calibration_tools():
        siril = Siril.from_config(cfg, workdir)   # raises SirilError when siril-cli is missing
        return SirilMasterBuilder(siril, library, workdir / "siril", cfg), SirilCalibrator(siril, workdir)

    library = DiskMasterRepository(Path(cfg["paths"]["library"]).expanduser())
    return PipelineServices(
        store=Workspace(workdir), provenance=ProvenanceStore(workdir), source=FilesystemFrameSource(),
        pixel_reader=FitsPixelReader(), pixel_stats=PixelStatsCache.from_config(workdir, cfg, reporter.progress),
        inventory_report=InventoryReportAdapter(), qa_report=QAReportAdapter(),
        masters_report=MastersReportAdapter(), repo=library,
        references=PixInsightMasters.from_config(cfg, source.resolve()), calibration_tools=calibration_tools,
        reporter=reporter, start_log=step_log(workdir, cfg).start)


def run_pipeline(source: Path, cfg: dict[str, Any], workdir: Path, from_step: str = "inventory",
                 sessions: list[str] | None = None, filters: list[str] | None = None,
                 force: bool = False) -> dict[str, Any]:
    """« varstar run »."""
    pipeline_use_case.check_step(from_step)
    workdir.mkdir(parents=True, exist_ok=True)
    return pipeline_use_case.run_pipeline(source, cfg, pipeline_services(source, cfg, workdir), from_step,
                                          sessions, filters, force)
