"""FITS file access for raw frames: primary header reading, ADU scale on file pixels.

Raw files are opened read-only; nothing here ever writes next to them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..domain.headers import frame_from_header
from ..domain.model import Frame
from ..domain.units import detect_adu_scale


def read_frame(path: Path, root: Path) -> Frame:
    """Read only the primary header (no pixel data)."""
    header = fits.getheader(path, 0)
    return frame_from_header(path, header, root)


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


class FitsPixelReader:
    """``PixelReader`` on raw FITS files (read-only)."""

    def adu_scale(self, frame: Frame) -> int:
        return detect_adu_scale_file(frame.path)
