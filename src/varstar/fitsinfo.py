"""Compatibility shim (refactoring step 2): header interpretation moved to
:mod:`varstar.domain.headers`, file reading to :mod:`varstar.adapters.fits_io`. Removed at step 8.
"""

from .adapters.fits_io import read_frame  # noqa: F401
from .domain.headers import (folder_label_from_path, frame_from_header, normalize_imagetyp,  # noqa: F401
                             parse_date_obs)
from .domain.model import BIAS, DARK, DARKFLAT, FLAT, LIGHT, UNKNOWN, Frame, SensorKey  # noqa: F401
