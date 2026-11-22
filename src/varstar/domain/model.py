"""Domain model: frames, calibration series, light groups, matches, inventory.

Pure data and derived properties (no I/O). Levels: A (full calibration), B (fallback,
explained), C (degraded); a group is as good as its worst component.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

LIGHT, DARK, FLAT, DARKFLAT, BIAS, UNKNOWN = "LIGHT", "DARK", "FLAT", "DARKFLAT", "BIAS", "UNKNOWN"


class SensorKey(NamedTuple):
    instrument: str
    gain: float | None
    offset: float | None
    xbin: int
    ybin: int
    naxis1: int
    naxis2: int

    def label(self) -> str:
        off = "?" if self.offset is None else f"{self.offset:g}"
        gain = "?" if self.gain is None else f"{self.gain:g}"
        return (f"{self.instrument} | gain {gain} | offset {off} | "
                f"bin {self.xbin}x{self.ybin} | {self.naxis1}x{self.naxis2}")


@dataclass
class Frame:
    path: Path
    rel_path: str
    imagetyp: str
    imagetyp_raw: str | None
    exptime: float | None
    gain: float | None
    offset: float | None
    xbin: int
    ybin: int
    naxis1: int
    naxis2: int
    bitpix: int
    instrument: str
    ccd_temp: float | None
    set_temp: float | None
    filter: str | None
    date_obs: dt.datetime | None
    object: str | None = None
    ra: float | None = None
    dec: float | None = None
    focallen: float | None = None
    xpixsz: float | None = None
    egain: float | None = None
    sitelat: float | None = None
    sitelong: float | None = None
    siteelev: float | None = None
    airmass: float | None = None
    telescope: str | None = None
    swcreate: str | None = None
    focpos: float | None = None
    foctemp: float | None = None
    centalt: float | None = None
    cloudcvr: float | None = None
    ambtemp: float | None = None
    dewpoint: float | None = None
    humidity: float | None = None
    folder_label: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def sensor_key(self) -> SensorKey:
        return SensorKey(self.instrument, self.gain, self.offset, self.xbin, self.ybin,
                         self.naxis1, self.naxis2)

    def session_date(self, offset_hours: float = -12.0) -> dt.date | None:
        if self.date_obs is None:
            return None
        return (self.date_obs + dt.timedelta(hours=offset_hours)).date()


LEVELS = ("A", "B", "C")


def worst(*levels: str | None) -> str:
    present = [lv for lv in levels if lv]
    return max(present, key=LEVELS.index) if present else "C"


# --------------------------------------------------------------------------- data model


@dataclass
class Anomaly:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    files: list[str] = field(default_factory=list)


@dataclass
class FrameSet:
    """A calibration series: same type, sensor key, exposure (and filter), contiguous in time."""

    id: str
    kind: str
    sensor_key: SensorKey
    exptime: float
    filter: str | None
    frames: list[Frame]

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> dt.datetime:
        return min(f.date_obs for f in self.frames)

    @property
    def end(self) -> dt.datetime:
        return max(f.date_obs for f in self.frames)

    @property
    def mid(self) -> dt.datetime:
        return self.start + (self.end - self.start) / 2

    @property
    def mean_temp(self) -> float | None:
        temps = [f.ccd_temp for f in self.frames if f.ccd_temp is not None]
        return statistics.fmean(temps) if temps else None

    @property
    def folder_labels(self) -> list[str]:
        return sorted({f.folder_label for f in self.frames if f.folder_label})

    @property
    def folders(self) -> list[str]:
        return sorted({str(Path(f.rel_path).parent) for f in self.frames})


@dataclass
class Match:
    level: str | None  # None = not applicable
    sets: list[FrameSet] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    borrowed_night: bool = False  # darks of another night (cannot be redone afterwards)

    @property
    def n_frames(self) -> int:
        return sum(s.n for s in self.sets)


@dataclass
class LightGroup:
    session: dt.date
    label: str | None
    filter: str
    sensor_key: SensorKey
    exptime: float
    frames: list[Frame]
    dark: Match | None = None
    flat: Match | None = None
    darkflat: Match | None = None

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> dt.datetime:
        return min(f.date_obs for f in self.frames)

    @property
    def end(self) -> dt.datetime:
        return max(f.date_obs for f in self.frames)

    @property
    def mean_temp(self) -> float | None:
        temps = [f.ccd_temp for f in self.frames if f.ccd_temp is not None]
        return statistics.fmean(temps) if temps else None

    @property
    def mean_set_temp(self) -> float | None:
        temps = [f.set_temp for f in self.frames if f.set_temp is not None]
        return statistics.fmean(temps) if temps else None

    @property
    def level(self) -> str:
        return worst(self.dark.level if self.dark else "C",
                     self.flat.level if self.flat else "C",
                     self.darkflat.level if self.darkflat else None)

    @property
    def reasons(self) -> list[str]:
        out: list[str] = []
        for m in (self.dark, self.flat, self.darkflat):
            if m:
                out.extend(m.reasons)
        return out


@dataclass
class SensorInfo:
    key: SensorKey
    n_frames: int
    egain: float | None
    adu_scale: int | None
    adu_scale_source: str | None
    xpixsz: float | None
    focallen: float | None

    @property
    def effective_gain(self) -> float | None:
        if self.egain is None or not self.adu_scale:
            return None
        return self.egain / self.adu_scale

    @property
    def saturation_adu(self) -> int | None:
        if not self.adu_scale:
            return None
        return (65535 // self.adu_scale) * self.adu_scale

    @property
    def pixel_scale_arcsec(self) -> float | None:
        # N.I.N.A. writes XPIXSZ already multiplied by the binning factor.
        if not self.xpixsz or not self.focallen:
            return None
        return 206.264806 * self.xpixsz / self.focallen


@dataclass
class DarkTodo:
    kind: str  # "dark" | "darkflat"
    sensor_key: SensorKey
    exptime: float
    set_temp: float | None
    mean_temp: float | None
    filters: list[str]
    sessions: list[str]
    current_level: str
    reason: str


@dataclass
class Inventory:
    root: Path
    config: dict[str, Any]
    frames: list[Frame]
    light_groups: list[LightGroup]
    calibration_sets: dict[str, list[FrameSet]]
    sensors: list[SensorInfo]
    anomalies: list[Anomaly]
    ignored_dirs: list[str]
    empty_dirs: list[str]
    other_extensions: dict[str, int]
    created: dt.datetime

    def session_label(self, session: dt.date) -> str:
        labels = {g.label for g in self.light_groups if g.session == session and g.label}
        return f"{session.isoformat()} ({', '.join(sorted(labels))})" if labels else session.isoformat()

    def levels_by_session_filter(self) -> dict[tuple[dt.date, str], str]:
        out: dict[tuple[dt.date, str], str] = {}
        for g in self.light_groups:
            key = (g.session, g.filter)
            out[key] = worst(out.get(key), g.level) if key in out else g.level
        return out

    def counts_by_type(self) -> Counter:
        return Counter(f.imagetyp for f in self.frames)


@dataclass
class SourceScan:
    """What was found under a source folder: frames read from the headers and folder facts."""

    root: Path
    frames: list[Frame]
    unreadable: list[str] = field(default_factory=list)   # "<relative path> (<exception>)"
    ignored_dirs: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    other_extensions: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AduScaleReading:
    """ADU scale measured on the pixels of one sample frame (or why it could not be)."""

    sample: str                 # relative path of the frame read
    scale: int | None = None
    error: str | None = None
