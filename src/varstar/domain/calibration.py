"""Calibration decisions (strategy B, SPEC §4): masters, rejection, dark mode, masks, identities.

Everything here is pure: file identities (path, size, mtime) are measured by the adapters and
passed in as ``FileStat``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .model import DARK, DARKFLAT, Frame, FrameSet, Inventory, LightGroup
from .qa.pixels import robust_sigma

MASTER_ALGO_VERSION = 1
CALIB_VERSION = 1
SATURATION_ADU = 65532

DARK_KINDS = ("dark", "darkflat")
KINDS = ("dark", "darkflat", "flat")


class PipelineError(RuntimeError):
    """A problem explained to the user (French message), which stops the step."""


@dataclass(frozen=True)
class FileStat:
    """Identity of a file on disk: resolved path, size and modification time (ns)."""

    resolved: str
    size: int
    mtime_ns: int


# --------------------------------------------------------------------------- rejection


@dataclass(frozen=True)
class Rejection:
    code: str  # Siril letter
    low: float
    high: float
    name: str  # French label

    def siril_args(self) -> str:
        return f"rej {self.code} {self.low:g} {self.high:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "low": self.low, "high": self.high, "name": self.name}


def choose_rejection(n: int, cfg: Mapping[str, Any]) -> Rejection:
    """SPEC §4.2: rejection algorithm depends on the number of frames."""
    m = cfg["masters"]
    lo, hi = float(m["sigma_low"]), float(m["sigma_high"])
    if n < 8:
        return Rejection("p", float(m.get("percentile_low", 0.2)), float(m.get("percentile_high", 0.1)),
                         "rejet par percentile")
    if n < 15:
        return Rejection("s", lo, hi, "sigma clipping")
    if n <= 25:
        return Rejection("w", lo, hi, "winsorized sigma clipping")
    return Rejection("l", lo, hi, "linear fit clipping")


# --------------------------------------------------------------------------- master identity


def frame_identity(stat: FileStat) -> str:
    return f"{stat.resolved}|{stat.size}|{stat.mtime_ns}"


def dark_params(rej: Rejection) -> dict[str, Any]:
    return {"rejection": [rej.code, rej.low, rej.high], "norm": "none"}


def flat_params(rej: Rejection, darkflat_id: str) -> dict[str, Any]:
    return {"rejection": [rej.code, rej.low, rej.high], "norm": "mul", "darkflat": darkflat_id}


def compute_master_id(kind: str, stats: Iterable[FileStat], params: Mapping[str, Any]) -> str:
    """Hash of the exact inputs: identical source files and parameters → same master, whatever the target."""
    h = hashlib.sha256()
    h.update(json.dumps({"kind": kind, "algo": MASTER_ALGO_VERSION, **params}, sort_keys=True).encode())
    for ident in sorted(frame_identity(s) for s in stats):
        h.update(ident.encode() + b"\n")
    return h.hexdigest()[:16]


def master_label(kind: str, frames: Sequence[Frame], extra: str = "") -> str:
    f = frames[0]
    gain = "?" if f.gain is None else f"{f.gain:g}"
    off = "x" if f.offset is None else f"{f.offset:g}"
    return f"{kind.upper()}_g{gain}_o{off}_b{f.xbin}_{f.exptime:g}s{extra}"


def dark_label(kind: str, frames: Sequence[Frame]) -> str:
    temps = [f.ccd_temp for f in frames if f.ccd_temp is not None]
    return master_label(kind, frames, f"_T{np.mean(temps):.0f}" if temps else "")


def flat_label(frames: Sequence[Frame]) -> str:
    filt = frames[0].filter or "x"
    return master_label("flat", frames, f"_{filt}_{min(f.date_obs for f in frames):%Y%m%d}")


@dataclass
class MasterRecord:
    """A master of the library: ``dir`` holds master.fits, master.json (and unstable.fits for darks)."""

    id: str
    kind: str  # "dark" | "darkflat" | "flat"
    dir: Path
    meta: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.dir / "master.fits"

    @property
    def unstable_path(self) -> Path:
        return self.dir / "unstable.fits"


# --------------------------------------------------------------------------- frame statistics (masters)


def corner_boxes(shape: tuple[int, int], size: int = 256) -> dict[str, tuple[slice, slice]]:
    h, w = shape
    s = min(size, h // 4, w // 4)
    return {"haut_gauche": (slice(0, s), slice(0, s)), "haut_droit": (slice(0, s), slice(w - s, w)),
            "bas_gauche": (slice(h - s, h), slice(0, s)), "bas_droit": (slice(h - s, h), slice(w - s, w))}


def master_frame_stats(frame: Frame, data: np.ndarray) -> dict[str, Any]:
    """Statistics of one source frame of a master (Q2 consistency, flat levels)."""
    sub = data[::4, ::4]
    med = float(np.median(sub))
    corners = {k: float(np.median(data[sl]) - med) for k, sl in corner_boxes(data.shape).items()}
    return {"file": frame.rel_path, "date_obs": frame.date_obs.isoformat() if frame.date_obs else None,
            "ccd_temp": frame.ccd_temp, "median": med, "robust_sigma": robust_sigma(sub),
            "frac_saturation": med / SATURATION_ADU, "corners": corners}


class TemporalNoise:
    """Per-pixel temporal noise of a dark series, accumulated frame by frame.

    The per-pixel standard deviation is computed with the largest value of each pixel removed (a
    single cosmic-ray hit does not make a pixel 'unstable'). The single-frame noise comes from the
    difference of successive frames.
    """

    def __init__(self) -> None:
        self.n = 0
        self.s1 = self.s2 = self.mx = self.prev = None
        self.diffs_sigma: list[float] = []

    def add(self, data: np.ndarray) -> None:
        d64 = data.astype(np.float64)
        if self.s1 is None:
            self.s1, self.s2, self.mx = d64.copy(), d64 * d64, data.copy()
        else:
            self.s1 += d64
            self.s2 += d64 * d64
            np.maximum(self.mx, data, out=self.mx)
        if self.prev is not None and len(self.diffs_sigma) < 5:
            self.diffs_sigma.append(robust_sigma((data - self.prev)[::2, ::2]) / np.sqrt(2))
        self.prev = data
        self.n += 1

    def result(self) -> dict[str, Any]:
        """``pixel_std`` (≥ 3 frames) and ``sigma_single`` (≥ 2 frames)."""
        extra: dict[str, Any] = {}
        n = self.n
        if n >= 3:
            mxd = self.mx.astype(np.float64)
            s1 = self.s1 - mxd
            s2 = self.s2 - mxd * mxd
            m = s1 / (n - 1)
            var = np.maximum(s2 / (n - 1) - m * m, 0.0) * (n - 1) / max(n - 2, 1)
            extra["pixel_std"] = np.sqrt(var).astype(np.float32)
            extra["sigma_single"] = float(np.median(self.diffs_sigma)) if self.diffs_sigma else None
        elif n == 2:
            extra["sigma_single"] = float(np.median(self.diffs_sigma)) if self.diffs_sigma else None
        return extra


def unstable_mask(pixel_std: np.ndarray, k: float) -> tuple[np.ndarray, float]:
    """Pixels whose temporal noise exceeds k × the median one (RTS); returns (uint8 mask, median std)."""
    med_std = float(np.median(pixel_std[::4, ::4]))
    return (pixel_std > k * med_std).astype(np.uint8), med_std


def flat_level_warnings(stats: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> list[str]:
    m = cfg["masters"]
    lo, hi = float(m["flat_level_min_frac"]), float(m["flat_level_max_frac"])
    return [s["file"] for s in stats if not lo <= s["frac_saturation"] <= hi]


def dark_consistency(frame_stats: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Q2: compare the dark sub-groups by folder (e.g. INTENSITY_dark / INTENSITY_light)."""
    by_folder: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for s in frame_stats:
        by_folder[str(PurePath(s["file"]).parent)].append(s)
    rows = []
    for folder, st in sorted(by_folder.items()):
        corners = {k: float(np.mean([x["corners"][k] for x in st])) for k in st[0]["corners"]}
        rows.append({"folder": folder, "n": len(st), "median_adu": float(np.mean([x["median"] for x in st])),
                     "robust_sigma_adu": float(np.mean([x["robust_sigma"] for x in st])),
                     "corner_excess_adu": corners,
                     "mean_ccd_temp": float(np.mean([x["ccd_temp"] for x in st if x["ccd_temp"] is not None]))})
    return rows


def crosscheck_stats(kind: str, ours: np.ndarray, theirs: np.ndarray) -> dict[str, Any]:
    """Compare one of our masters with the PixInsight one (also tried flipped vertically)."""
    if theirs.shape != ours.shape:
        return {"error": f"dimensions différentes {theirs.shape} / {ours.shape}"}
    a = ours[::2, ::2]
    best = None
    for flipped in (False, True):
        b = (theirs[::-1] if flipped else theirs)[::2, ::2]
        if kind == "flat":
            r = (a / np.median(a)) / (b / np.median(b)) - 1.0
            stat = {"robust_sigma_ratio_pct": 100 * robust_sigma(r), "median_ratio_pct": 100 * float(np.median(r))}
            score = stat["robust_sigma_ratio_pct"]
        else:
            d = a - b
            stat = {"median_ours_adu": float(np.median(a)), "median_pi_adu": float(np.median(b)),
                    "median_diff_adu": float(np.median(d)), "robust_sigma_diff_adu": robust_sigma(d)}
            score = stat["robust_sigma_diff_adu"]
        if best is None or score < best[0]:
            best = (score, flipped, stat)
    return {"flipped": best[1], **best[2]}


def master_summary(rec: MasterRecord) -> dict[str, Any]:
    """Entry of a master in plan.json."""
    m = rec.meta
    return {"kind": rec.kind, "label": m["label"], "dir": str(rec.dir), "n": m["n_frames"],
            "rejection": m["rejection"]["name"], "median_adu": m["median_adu"], "exptime": m["exptime"],
            "filter": m.get("filter"), "sigma_single_frame_adu": m.get("sigma_single_frame_adu"),
            "n_unstable": m.get("n_unstable"), "level_min_frac": m.get("level_min_frac"),
            "level_max_frac": m.get("level_max_frac")}


# --------------------------------------------------------------------------- light groups


def group_key(g: LightGroup) -> str:
    return f"{g.session.isoformat()}_{g.filter}_{g.exptime:g}s"


def select_groups(inv: Inventory, sessions: list[str] | None, filters: list[str] | None) -> list[LightGroup]:
    out = []
    for g in inv.light_groups:
        if sessions and not ({g.session.isoformat(), g.label or ""} & set(sessions)):
            continue
        if filters and g.filter not in filters:
            continue
        out.append(g)
    if sessions:
        known = {g.session.isoformat() for g in inv.light_groups} | {g.label for g in inv.light_groups if g.label}
        unknown = set(sessions) - known
        if unknown:
            raise PipelineError(f"Session(s) inconnue(s) : {', '.join(sorted(unknown))}")
    return out


def frames_of(sets: Iterable[FrameSet]) -> list[Frame]:
    return sorted((f for s in sets for f in s.frames), key=lambda f: f.date_obs)


def dark_mode(dark_sets: Sequence[FrameSet], exptime: float, tolerance_s: float) -> str:
    """'direct' when every dark has the exposure of the lights, else 'scaled' (fallback, level B)."""
    return "scaled" if any(abs(s.exptime - exptime) > tolerance_s for s in dark_sets) else "direct"


def offset_dark_sets(inv: Inventory, g: LightGroup) -> tuple[str, list[FrameSet]]:
    """Shortest darks of the sensor: their median estimates the uniform offset for dark scaling."""
    pool = [s for kind in (DARKFLAT, DARK) for s in inv.calibration_sets[kind] if s.sensor_key == g.sensor_key]
    if not pool:
        raise PipelineError("aucun dark court pour estimer l'offset")
    shortest = min(s.exptime for s in pool)
    sets = [s for s in pool if s.exptime == shortest]
    return ("darkflat" if sets[0].kind == DARKFLAT else "dark"), sets


def darkflat_kind(sets: Sequence[FrameSet]) -> str:
    return "darkflat" if sets[0].kind == DARKFLAT else "dark"


def mask_ids(dark_id: str | None, flat_id: str | None, masks_hash: str) -> tuple[str, str | None]:
    """File names (in masks/) of the bad pixel map of a dark/flat pair and of the dark's amp glow mask."""
    bpm = f"bpm_{dark_id or 'nodark'}_{flat_id or 'noflat'}_{masks_hash}.fits"
    ampglow = f"ampglow_{dark_id}_{masks_hash}.fits" if dark_id else None
    return bpm, ampglow


# --------------------------------------------------------------------------- calibrated frames


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", PurePath(name).stem).strip("_")


def calibrated_subdir(session: str, label: str | None, filt: str) -> str:
    """calibrated/<session>[_<label>]/<filter> (relative to the work directory)."""
    return f"calibrated/{session}{'_' + label if label else ''}/{filt}"


def calibrated_name(raw: str) -> str:
    return f"{safe_stem(PurePath(raw).name)}.fits"


def calibration_fingerprint(group: Mapping[str, Any], raws: Sequence[tuple[str, FileStat]]) -> str:
    """Changes when a master, a mask, the level or a raw file of the group changes."""
    h = hashlib.sha256()
    payload = {k: group.get(k) for k in ("level", "dark", "flat", "darkflat", "bpm", "ampglow")}
    h.update(json.dumps({"v": CALIB_VERSION, **payload}, sort_keys=True, default=str).encode())
    for raw, st in raws:
        h.update(f"{raw}|{st.size}|{st.mtime_ns}\n".encode())
    return h.hexdigest()


@dataclass
class CalibrationSensor:
    """Sensor facts written into the calibrated headers."""

    adu_scale: int | None = None
    effective_gain: float | None = None

    @classmethod
    def of(cls, inv: Inventory) -> "CalibrationSensor":
        s = inv.sensors[0] if inv.sensors else None
        return cls(s.adu_scale if s else None, s.effective_gain if s else None)
