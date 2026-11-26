"""Compatibility module (refactoring step 3).

The rules moved to :mod:`varstar.domain.inventory`, the use case to :mod:`varstar.application.inventory`.
``build_inventory(root, cfg)`` is kept as a wrapper wiring the file adapters until the composition
root (``varstar.cli.bootstrap``). Removed at step 8.
"""

from __future__ import annotations

from pathlib import Path

from .domain.inventory import (adu_scale_samples, build_sets, darks_todo, folder_consistency,  # noqa: F401
                               match_darkflats, match_darks, match_flats, nearest_night, set_night,
                               usable_frames)
from .domain.model import (BIAS, DARK, DARKFLAT, FLAT, LEVELS, LIGHT, UNKNOWN, Anomaly, DarkTodo,  # noqa: F401
                           Frame, FrameSet, Inventory, LightGroup, Match, SensorInfo, SensorKey, worst)
from .domain.settings import Settings


def build_inventory(root: Path, cfg: Settings, exclude: list[Path] | None = None) -> Inventory:
    from .cli import bootstrap
    return bootstrap.load_inventory(root, cfg, exclude)
