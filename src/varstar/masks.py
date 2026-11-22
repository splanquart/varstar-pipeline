"""Compatibility shim (refactoring step 1): moved to :mod:`varstar.domain.masks`. Removed at step 8."""

from .domain.masks import (BIT_NAMES, BIT_NAMES_FR, COLD, DEAD, HOT, SATURATED, UNSTABLE,  # noqa: F401
                           VERY_HOT, MaskResult, block_median, build_amp_glow, build_bpm, robust_sigma,
                           upsample_nearest)
