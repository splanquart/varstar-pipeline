"""Legacy entry points of the steps (removed at step 8): see ``varstar.cli.bootstrap``."""

from __future__ import annotations

from pathlib import Path

from .adapters.console import setup_logging  # noqa: F401
from .application.inventory import InventoryResult
from .application.pipeline import STEPS  # noqa: F401
from .domain.settings import Settings


def run_inventory(source: Path, cfg: Settings, workdir: Path, force: bool = False) -> InventoryResult:
    from .cli import bootstrap
    return bootstrap.run_inventory(source, cfg, workdir, force)
