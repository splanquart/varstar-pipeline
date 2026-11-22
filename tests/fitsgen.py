"""Synthetic N.I.N.A.-like FITS writers shared by the tests (no dependency on the real dataset)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
from astropy.io import fits


def write_frame(path: Path, imagetyp: str, exptime: float, date_obs: str, *, filt: str | None = None,
                temp: float = -12.0, gain: int = 120, xbin: int = 2, shape=(12, 16), scale: int = 4,
                offset: int | None = None, instrument: str = "ASI Camera (1) (ASCO",
                extra: dict | None = None, data: np.ndarray | None = None) -> Path:
    rng = np.random.default_rng(abs(hash(str(path))) % 2**32)
    if data is None:
        data = (rng.integers(100, 2000, size=shape) * scale).astype(np.uint16)
    hdu = fits.PrimaryHDU(data)
    h = hdu.header
    h["IMAGETYP"] = imagetyp
    h["EXPOSURE"] = exptime
    h["EXPTIME"] = exptime
    h["DATE-OBS"] = date_obs
    h["XBINNING"] = xbin
    h["YBINNING"] = xbin
    h["GAIN"] = gain
    if offset is not None:
        h["OFFSET"] = offset
    h["EGAIN"] = 1.00224268436432
    h["XPIXSZ"] = 4.63
    h["INSTRUME"] = instrument
    h["SET-TEMP"] = -12.0
    h["CCD-TEMP"] = temp
    h["FOCALLEN"] = 274.0
    h["FILTER"] = filt if filt is not None else "Dark"
    h["OBJECT"] = "NGC 1499" if imagetyp == "LIGHT" else ""
    for k, v in (extra or {}).items():
        h[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def series(root: Path, folder: str, imagetyp: str, exptime: float, start: str, n: int, *,
           step_s: float = 60, prefix: str = "NGC 1499", make=None, **kw) -> list[Path]:
    """`make(i)` may return the pixel data (uint16) of frame i."""
    t0 = dt.datetime.fromisoformat(start)
    out = []
    for i in range(n):
        t = t0 + dt.timedelta(seconds=i * step_s)
        name = f"{t:%Y-%m-%d_%H-%M-%S}_{prefix}_{kw.get('filt') or 'Dark'}_{exptime:.2f}s_120_{i:04d}.fits"
        data = make(i) if make else None
        out.append(write_frame(root / folder / name, imagetyp, exptime, t.isoformat(timespec="milliseconds"),
                               data=data, **kw))
    return out


# --------------------------------------------------------------------------- two nights with pixel noise

QA_SHAPE = (128, 160)


def noise(level: float, sigma: float, seed: int, scale: int = 4, extra=None, shape=QA_SHAPE) -> np.ndarray:
    """Gaussian frame around ``level`` ADU, quantised like the ASI294MM (14 bits × ``scale``)."""
    rng = np.random.default_rng(seed)
    adu14 = rng.normal(level / scale, sigma / scale, shape)
    if extra is not None:
        adu14 = adu14 + extra / scale
    return (np.clip(np.round(adu14), 0, 16383) * scale).astype(np.uint16)


def two_nights(root: Path, level2: float = 516.0, leak: bool = False, temp: float = -12.0) -> Path:
    """sept02-like night with darks at 516 ADU and a sept06-like night with darks at ``level2``.

    ``leak``: the second night's darks get extra photons (Poisson noise at 0.25 e-/ADU) instead.
    """
    series(root, "light/H/SESSION_sept02", "LIGHT", 300, "2021-09-02T23:30:00", 3, step_s=600, filt="H",
           shape=QA_SHAPE, make=lambda i: noise(600, 12, 100 + i))
    series(root, "light/H/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:04:00", 3, step_s=600, filt="H",
           shape=QA_SHAPE, make=lambda i: noise(600, 12, 200 + i), temp=temp)
    series(root, "dark/a", "DARK", 300, "2021-09-02T20:00:00", 5, step_s=310, prefix="", shape=QA_SHAPE,
           make=lambda i: noise(516, 12, 300 + i))

    def night2(i):
        rng = np.random.default_rng(400 + i)
        if leak:
            light = rng.poisson(25 * 0.25, QA_SHAPE) / 0.25
            return noise(516, 12, 400 + i, extra=light)
        return noise(level2, 12, 400 + i)

    series(root, "dark/b", "DARK", 300, "2021-09-07T02:00:00", 5, step_s=310, prefix="", shape=QA_SHAPE, make=night2)
    return root
