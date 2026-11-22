"""ADU scale detection on a FITS file (the pure detection lives in varstar.domain.units)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from .domain.units import MAX_SCALE, detect_adu_scale  # noqa: F401


def detect_adu_scale_file(path: Path) -> int:
    # Raw (unscaled) integers keep memory mapping possible; BZERO = 32768 is a
    # multiple of every candidate scale, so it is added back as an integer.
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        hdu = hdul[0]
        bzero = hdu.header.get("BZERO", 0)
        bscale = hdu.header.get("BSCALE", 1)
        # Sub-sample rows/columns to keep it fast on 4144x2822 frames.
        sample = np.asarray(hdu.data[::5, ::3])
        if bscale != 1 or float(bzero) != int(bzero):
            return detect_adu_scale(sample * bscale + bzero)
        return detect_adu_scale(sample.astype(np.int64) + int(bzero))
