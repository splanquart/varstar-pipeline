"""Compatibility shim (refactoring step 2): pure detection in :mod:`varstar.domain.units`,
detection on a file in :mod:`varstar.adapters.fits_io`. Removed at step 8.
"""

from .adapters.fits_io import detect_adu_scale_file  # noqa: F401
from .domain.units import MAX_SCALE, detect_adu_scale  # noqa: F401
