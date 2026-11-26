"""Legacy entry point of the pipeline (removed at step 8): see ``varstar.application.pipeline`` and
``varstar.cli.bootstrap``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters.reporting.masters_markdown import masters_markdown  # noqa: F401
from .application.pipeline import IMPLEMENTED, STEPS  # noqa: F401
from .domain.calibration import PipelineError, group_key, select_groups  # noqa: F401
from .domain.settings import Settings


def run_pipeline(source: Path, cfg: Settings, workdir: Path, from_step: str = "inventory",
                 sessions: list[str] | None = None, filters: list[str] | None = None,
                 force: bool = False) -> dict[str, Any]:
    from .cli import bootstrap
    return bootstrap.run_pipeline(source, cfg, workdir, from_step, sessions, filters, force)
