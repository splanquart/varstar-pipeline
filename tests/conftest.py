"""Synthetic N.I.N.A.-like FITS files for unit tests (no dependency on the real dataset)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from varstar.config import load_config


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


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def complete_session(tmp_path: Path) -> Path:
    """One complete night (sept06-like): H/S lights, flats next morning, darks, dark-flats."""
    root = tmp_path / "YEAR"
    series(root, "light/H/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:04:00", 3, step_s=600, filt="H")
    series(root, "light/S/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:09:00", 3, step_s=600, filt="S")
    series(root, "flat/SESSION_sept06", "FLAT", 1.13, "2021-09-07T13:21:00", 3, step_s=2, filt="H")
    series(root, "flat/SESSION_sept06", "FLAT", 2.62, "2021-09-07T13:22:00", 3, step_s=2, filt="S")
    series(root, "dark/darklight/300s", "DARK", 300, "2021-09-07T02:00:00", 4, step_s=310, prefix="")
    series(root, "dark/darkflat", "DARKFLAT", 1.13, "2021-09-08T18:25:00", 3, step_s=2, prefix="")
    series(root, "dark/darkflat", "DARKFLAT", 2.62, "2021-09-07T13:28:00", 3, step_s=3, prefix="")
    return root
