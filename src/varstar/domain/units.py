"""ADU scale detection (e.g. 14-bit data stored as 16-bit multiples of 4)."""

from __future__ import annotations

import numpy as np

MAX_SCALE = 16


def detect_adu_scale(data: np.ndarray, max_samples: int = 2_000_000) -> int:
    """Return the largest power of two (<= 16) dividing every pixel value.

    N.I.N.A. stores the ASI294MM 14-bit samples left-shifted to 16 bits, so all
    values are multiples of 4 and the effective gain is EGAIN / 4.
    """
    flat = np.asarray(data).ravel()
    if flat.size > max_samples:
        step = flat.size // max_samples + 1
        flat = flat[::step]
    if not np.issubdtype(flat.dtype, np.integer):
        if not np.all(np.isfinite(flat)) or not np.all(flat == np.round(flat)):
            return 1
    values = flat.astype(np.int64)
    values = values[values != 0]
    if values.size == 0:
        return 1
    g = int(np.gcd.reduce(np.abs(values)))
    scale = 1
    while scale * 2 <= MAX_SCALE and g % (scale * 2) == 0:
        scale *= 2
    return scale
