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
