"""In-memory builders for domain objects (no file, no FITS): tests of the domain layer.

Defaults describe the reference setup (ASI294MM, gain 120, bin 2, −12 °C, NGC 1499 2021).
Every field can be overridden by keyword.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from varstar.config import load_config
from varstar.domain.model import (DARK, FLAT, LIGHT, Frame, FrameSet, Inventory, LightGroup, Match, SensorInfo,
                                  SensorKey)

T0 = dt.datetime(2021, 9, 6, 23, 0, 0)
SENSOR = SensorKey("ASI Camera (1) (ASCO", 120.0, None, 2, 2, 4144, 2822)
NINA_DEFAULTS = {"egain": 1.00224268436432, "xpixsz": 4.63, "focallen": 274.0}


def make_settings(**overrides: Any) -> dict[str, Any]:
    """Default configuration with section overrides, e.g. ``make_settings(matching={...})``."""
    return load_config(overrides=overrides)


def make_frame(imagetyp: str = LIGHT, date_obs: dt.datetime = T0, *, exptime: float = 300.0,
               filter: str | None = "auto", ccd_temp: float | None = -12.0, set_temp: float | None = -12.0,
               rel_path: str | None = None, folder_label: str | None = None, sensor: SensorKey = SENSOR,
               **extra: Any) -> Frame:
    if filter == "auto":   # lights and flats get a filter, darks none
        filter = "H" if imagetyp in (LIGHT, FLAT) else None
    for key, value in NINA_DEFAULTS.items():
        extra.setdefault(key, value)
    stamp = f"{date_obs:%Y-%m-%d_%H-%M-%S}" if date_obs else "nodate"
    rel = rel_path or f"{imagetyp.lower()}/{stamp}_{filter or 'Dark'}_{exptime:g}s.fits"
    return Frame(path=Path("/raw") / rel, rel_path=rel, imagetyp=imagetyp, imagetyp_raw=imagetyp,
                 exptime=exptime, gain=sensor.gain, offset=sensor.offset, xbin=sensor.xbin, ybin=sensor.ybin,
                 naxis1=sensor.naxis1, naxis2=sensor.naxis2, bitpix=16, instrument=sensor.instrument,
                 ccd_temp=ccd_temp, set_temp=set_temp, filter=filter, date_obs=date_obs,
                 folder_label=folder_label, **extra)


def make_frames(n: int, imagetyp: str = LIGHT, start: dt.datetime = T0, step_s: float = 60.0,
                **kw: Any) -> list[Frame]:
    return [make_frame(imagetyp, start + dt.timedelta(seconds=i * step_s), **kw) for i in range(n)]


def make_frameset(kind: str = DARK, n: int = 5, start: dt.datetime = T0, *, exptime: float = 300.0,
                  filter: str | None = None, step_s: float | None = None, id: str | None = None,
                  **kw: Any) -> FrameSet:
    frames = make_frames(n, kind, start, step_s if step_s is not None else exptime + 10, exptime=exptime,
                         filter=filter, **kw)
    sid = id or f"{kind}-{exptime:g}s-{start:%Y%m%d}"
    return FrameSet(sid, kind, frames[0].sensor_key, exptime, filter if kind == FLAT else None, frames)


def make_light_group(n: int = 3, start: dt.datetime = T0, *, filter: str = "H", exptime: float = 300.0,
                     label: str | None = "s1", dark: Match | None = None, flat: Match | None = None,
                     darkflat: Match | None = None, offset_hours: float = -12.0, **kw: Any) -> LightGroup:
    frames = make_frames(n, LIGHT, start, exptime + 100, exptime=exptime, filter=filter, folder_label=label, **kw)
    return LightGroup(frames[0].session_date(offset_hours), label, filter, frames[0].sensor_key, exptime, frames,
                      dark, flat, darkflat)


def make_sensor_info(*, egain: float | None = 1.00224268436432, adu_scale: int | None = 4,
                     xpixsz: float | None = 4.63, focallen: float | None = 274.0, n_frames: int = 10) -> SensorInfo:
    return SensorInfo(SENSOR, n_frames, egain, adu_scale, "light/x.fits", xpixsz, focallen)


def make_inventory(groups: list[LightGroup], *, frames: list[Frame] | None = None,
                   sets: dict[str, list[FrameSet]] | None = None, cfg: dict[str, Any] | None = None) -> Inventory:
    frames = frames if frames is not None else [f for g in groups for f in g.frames]
    return Inventory(Path("/raw"), cfg or make_settings(), frames, groups, sets or {}, [make_sensor_info()], [],
                     [], [], {}, dt.datetime(2026, 1, 1))


# --------------------------------------------------------------------------- sessions (in memory)


def series_frames(folder: str, imagetyp: str, exptime: float, start: str, n: int, *, step_s: float = 60,
                  filter: str | None = None, **kw: Any) -> list[Frame]:
    """In-memory twin of ``fitsgen.series``: same folders, same times, labels taken from the path."""
    from varstar.domain.headers import folder_label_from_path

    t0 = dt.datetime.fromisoformat(start)
    out = []
    for i in range(n):
        t = t0 + dt.timedelta(seconds=i * step_s)
        rel = f"{folder}/{t:%Y-%m-%d_%H-%M-%S}_{filter or 'Dark'}_{exptime:.2f}s_{i:04d}.fits"
        out.append(make_frame(imagetyp, t, exptime=exptime, filter=filter, rel_path=rel,
                              folder_label=folder_label_from_path(rel), **kw))
    return out


def sept06_frames() -> list[Frame]:
    """In-memory twin of the ``complete_session`` fixture: one complete night, level A everywhere."""
    from varstar.domain.model import DARKFLAT
    return [
        *series_frames("light/H/SESSION_sept06", LIGHT, 300, "2021-09-06T23:04:00", 3, step_s=600, filter="H"),
        *series_frames("light/S/SESSION_sept06", LIGHT, 300, "2021-09-06T23:09:00", 3, step_s=600, filter="S"),
        *series_frames("flat/SESSION_sept06", FLAT, 1.13, "2021-09-07T13:21:00", 3, step_s=2, filter="H"),
        *series_frames("flat/SESSION_sept06", FLAT, 2.62, "2021-09-07T13:22:00", 3, step_s=2, filter="S"),
        *series_frames("dark/darklight/300s", DARK, 300, "2021-09-07T02:00:00", 4, step_s=310),
        *series_frames("dark/darkflat", DARKFLAT, 1.13, "2021-09-08T18:25:00", 3, step_s=2),
        *series_frames("dark/darkflat", DARKFLAT, 2.62, "2021-09-07T13:28:00", 3, step_s=3),
    ]


def make_scan(frames: list[Frame], **kw: Any):
    from varstar.domain.model import SourceScan
    return SourceScan(Path("/raw"), frames, **kw)


# --------------------------------------------------------------------------- pixel statistics (in memory)

CORNERS = ("haut_gauche", "haut_droit", "bas_gauche", "bas_droit")


def make_frame_stats(level: float = 500.0, sigma: float = 12.0, *, corners: float | dict[str, float] = 0.0,
                     block_range: float = 0.0, p005: float | None = None, frac_saturated: float = 0.0,
                     adu_scale: int = 4, corner_npix: int = 100) -> dict[str, Any]:
    """Quick statistics of one raw frame, as ``domain.qa.pixels.frame_quick_stats`` returns them."""
    if not isinstance(corners, dict):
        corners = {k: float(corners) for k in CORNERS}
    return {"median": float(level), "robust_sigma": float(sigma),
            "p005": float(level - 3 * sigma if p005 is None else p005), "corners": corners,
            "corner_npix": corner_npix, "block_range": float(block_range), "frac_saturated": frac_saturated,
            "adu_scale": adu_scale, "shape": [2822, 4144]}


def stats_for(frames, **kw: Any) -> dict[str, dict[str, Any]]:
    """Same statistics for every frame, keyed like the ``PixelStats`` port (``str(frame.path)``)."""
    return {str(f.path): make_frame_stats(**kw) for f in frames}


def inventory_of(frames: list[Frame], cfg: dict[str, Any] | None = None, *, adu_scale: int | None = 4,
                 **scan: Any) -> Inventory:
    """Inventory built by the domain from frames in memory (ADU scale measured as ``adu_scale``)."""
    from varstar.domain.inventory import build_inventory
    from varstar.domain.model import AduScaleReading

    readings = None if adu_scale is None else {
        f.sensor_key: AduScaleReading(f.rel_path, scale=adu_scale) for f in frames}
    return build_inventory(make_scan(frames, **scan), cfg or make_settings(), readings, dt.datetime(2026, 1, 1))


def two_nights_frames(*, light_temp: float = -12.0) -> list[Frame]:
    """In-memory twin of ``fitsgen.two_nights``: darks the nights of 02/09 and 07/09, lights both nights."""
    return [
        *series_frames("light/H/SESSION_sept02", LIGHT, 300, "2021-09-02T23:30:00", 3, step_s=600, filter="H"),
        *series_frames("light/H/SESSION_sept06", LIGHT, 300, "2021-09-06T23:04:00", 3, step_s=600, filter="H",
                       ccd_temp=light_temp),
        *series_frames("dark/a", DARK, 300, "2021-09-02T20:00:00", 5, step_s=310),
        *series_frames("dark/b", DARK, 300, "2021-09-07T02:00:00", 5, step_s=310),
    ]
