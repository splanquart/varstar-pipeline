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


def make_settings(**overrides: Any) -> dict[str, Any]:
    """Default configuration with section overrides, e.g. ``make_settings(matching={...})``."""
    return load_config(overrides=overrides)


def make_frame(imagetyp: str = LIGHT, date_obs: dt.datetime = T0, *, exptime: float = 300.0,
               filter: str | None = "auto", ccd_temp: float | None = -12.0, set_temp: float | None = -12.0,
               rel_path: str | None = None, folder_label: str | None = None, sensor: SensorKey = SENSOR,
               **extra: Any) -> Frame:
    if filter == "auto":   # lights and flats get a filter, darks none
        filter = "H" if imagetyp in (LIGHT, FLAT) else None
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
