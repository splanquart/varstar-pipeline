"""Master darks / dark-flats / flats built with Siril and cached in a shared library.

Library layout (shared between targets and years):
    <library>/darks/<label>_<id>/master.fits   (+ master.json, unstable.fits for light darks)
    <library>/flats/<label>_<id>/master.fits   (+ master.json)

A master id is a hash of its exact inputs (source files, rejection parameters, algorithm
version, dark-flat master for flats): identical inputs -> reused, whatever the target.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .fitsinfo import Frame
from .imageio import NORM_SCALE, read_adu, read_siril_output, write_mask
from .siril import Siril, link_sequence

log = logging.getLogger("varstar")

MASTER_ALGO_VERSION = 1
SATURATION_ADU = 65532


# --------------------------------------------------------------------------- rejection choice


@dataclass(frozen=True)
class Rejection:
    code: str  # Siril letter
    low: float
    high: float
    name: str  # French label

    def siril_args(self) -> str:
        return f"rej {self.code} {self.low:g} {self.high:g}"


def choose_rejection(n: int, cfg: dict[str, Any]) -> Rejection:
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


# --------------------------------------------------------------------------- library


@dataclass
class MasterRecord:
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

    def load_adu(self) -> np.ndarray:
        return read_adu(self.path)[0]


def _frame_identity(f: Frame) -> str:
    st = f.path.stat()
    return f"{f.path.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def compute_master_id(kind: str, frames: list[Frame], params: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"kind": kind, "algo": MASTER_ALGO_VERSION, **params}, sort_keys=True).encode())
    for ident in sorted(_frame_identity(f) for f in frames):
        h.update(ident.encode() + b"\n")
    return h.hexdigest()[:16]


class Library:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def _subdir(self, kind: str) -> Path:
        return self.root / ("flats" if kind == "flat" else "darks")

    def find(self, kind: str, master_id: str) -> MasterRecord | None:
        base = self._subdir(kind)
        if not base.exists():
            return None
        for d in base.glob(f"*_{master_id}"):
            meta_path = d / "master.json"
            if (d / "master.fits").exists() and meta_path.exists():
                return MasterRecord(master_id, kind, d, json.loads(meta_path.read_text(encoding="utf-8")))
        return None

    def new_dir(self, kind: str, label: str, master_id: str) -> Path:
        d = self._subdir(kind) / f"{label}_{master_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d


# --------------------------------------------------------------------------- python-side statistics


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def _corner_boxes(shape: tuple[int, int], size: int = 256) -> dict[str, tuple[slice, slice]]:
    h, w = shape
    s = min(size, h // 4, w // 4)
    return {"haut_gauche": (slice(0, s), slice(0, s)), "haut_droit": (slice(0, s), slice(w - s, w)),
            "bas_gauche": (slice(h - s, h), slice(0, s)), "bas_droit": (slice(h - s, h), slice(w - s, w))}


def frame_stats(frames: list[Frame], temporal: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame statistics (Q2 consistency, flat levels) and optional per-pixel temporal noise.

    When `temporal` is set, the per-pixel standard deviation across frames is computed with the
    largest value of each pixel removed (a single cosmic-ray hit does not make a pixel 'unstable').
    """
    stats: list[dict[str, Any]] = []
    s1 = s2 = mx = None
    prev = None
    diffs_sigma: list[float] = []
    for f in frames:
        with fits.open(f.path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
        sub = data[::4, ::4]
        med = float(np.median(sub))
        corners = {k: float(np.median(data[sl]) - med) for k, sl in _corner_boxes(data.shape).items()}
        stats.append({"file": f.rel_path, "date_obs": f.date_obs.isoformat() if f.date_obs else None,
                      "ccd_temp": f.ccd_temp, "median": med, "robust_sigma": _robust_sigma(sub),
                      "frac_saturation": med / SATURATION_ADU, "corners": corners})
        if temporal:
            d64 = data.astype(np.float64)
            if s1 is None:
                s1, s2, mx = d64.copy(), d64 * d64, data.copy()
            else:
                s1 += d64
                s2 += d64 * d64
                np.maximum(mx, data, out=mx)
            if prev is not None and len(diffs_sigma) < 5:
                diffs_sigma.append(_robust_sigma((data - prev)[::2, ::2]) / np.sqrt(2))
            prev = data
    extra: dict[str, Any] = {}
    if temporal and len(frames) >= 3:
        n = len(frames)
        mxd = mx.astype(np.float64)
        s1 -= mxd
        s2 -= mxd * mxd
        m = s1 / (n - 1)
        var = np.maximum(s2 / (n - 1) - m * m, 0.0) * (n - 1) / max(n - 2, 1)
        extra["pixel_std"] = np.sqrt(var).astype(np.float32)
        extra["sigma_single"] = float(np.median(diffs_sigma)) if diffs_sigma else None
    elif temporal and len(frames) == 2:
        extra["sigma_single"] = float(np.median(diffs_sigma)) if diffs_sigma else None
    return stats, extra


# --------------------------------------------------------------------------- builders


@dataclass
class BuildContext:
    cfg: dict[str, Any]
    siril: Siril
    library: Library
    tmp_root: Path


def _finalize(rec_dir: Path, produced: Path, kind: str, meta: dict[str, Any], frames: list[Frame]) -> None:
    """Move Siril's result into the library with provenance keywords; write master.json."""
    with fits.open(produced, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    header["VSKIND"] = (kind.upper(), "varstar master type")
    header["VSID"] = (meta["id"], "varstar master id")
    header["VSNFRAM"] = (len(frames), "number of source frames")
    header["VSREJ"] = (meta["rejection"]["name"], "rejection algorithm")
    header["VSNORM"] = (NORM_SCALE, "ADU16 = value * VSNORM (Siril normalisation)")
    header["VSSIRIL"] = (meta["siril_version"], "Siril version")
    header["VSDATE"] = (meta["created"], "creation date")
    for f in frames:
        header.add_history(f"src {f.rel_path}"[:72])
    fits.PrimaryHDU(data, header=header).writeto(rec_dir / "master.fits", overwrite=True)
    (rec_dir / "master.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _label(kind: str, frames: list[Frame], extra: str = "") -> str:
    f = frames[0]
    gain = "?" if f.gain is None else f"{f.gain:g}"
    off = "x" if f.offset is None else f"{f.offset:g}"
    return f"{kind.upper()}_g{gain}_o{off}_b{f.xbin}_{f.exptime:g}s{extra}"


def build_dark_master(frames: list[Frame], kind: str, ctx: BuildContext, sources_root: Path) -> MasterRecord:
    """Master dark (kind='dark') or dark-flat (kind='darkflat'): mean with rejection, no normalisation."""
    rej = choose_rejection(len(frames), ctx.cfg)
    params = {"rejection": [rej.code, rej.low, rej.high], "norm": "none"}
    mid = compute_master_id(kind, frames, params)
    found = ctx.library.find(kind, mid)
    if found:
        log.info("Master %s %s déjà en bibliothèque (%s)", kind, mid, found.dir.name)
        return found

    temps = [f.ccd_temp for f in frames if f.ccd_temp is not None]
    label = _label(kind, frames, f"_T{np.mean(temps):.0f}" if temps else "")
    log.info("Construction du master %s %s (%d images, %s)", kind, label, len(frames), rej.name)
    work = ctx.tmp_root / f"{kind}_{mid}"
    if work.exists():
        shutil.rmtree(work)
    link_sequence([f.path for f in frames], work, "f")
    ctx.siril.run([f"stack f {rej.siril_args()} -nonorm -out=master -32b"], work, f"{kind}_{mid}")

    stats, extra = frame_stats(frames, temporal=(kind == "dark"))
    rec_dir = ctx.library.new_dir(kind, label, mid)
    master_adu = read_siril_output(work / "master.fits")[0]
    meta = {
        "id": mid, "kind": kind, "label": label, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "siril_version": ctx.siril.version(), "algo_version": MASTER_ALGO_VERSION,
        "rejection": {"code": rej.code, "low": rej.low, "high": rej.high, "name": rej.name},
        "n_frames": len(frames), "exptime": frames[0].exptime, "gain": frames[0].gain,
        "offset": frames[0].offset, "xbin": frames[0].xbin, "ybin": frames[0].ybin,
        "naxis1": frames[0].naxis1, "naxis2": frames[0].naxis2, "instrument": frames[0].instrument,
        "mean_ccd_temp": float(np.mean(temps)) if temps else None,
        "sources": [str(f.path) for f in frames], "sources_root": str(sources_root),
        "median_adu": float(np.median(master_adu[::4, ::4])),
        "robust_sigma_adu": _robust_sigma(master_adu[::4, ::4]),
        "sigma_single_frame_adu": extra.get("sigma_single"),
        "frame_stats": stats,
    }
    if "pixel_std" in extra:
        pstd = extra["pixel_std"]
        med_std = float(np.median(pstd[::4, ::4]))
        k = float(ctx.cfg["masks"].get("unstable_sigma_factor", 5.0))
        unstable = (pstd > k * med_std).astype(np.uint8)
        hdr = fits.Header()
        hdr["VSID"] = mid
        hdr["VSMEDSTD"] = (med_std, "median per-pixel temporal std (ADU16)")
        hdr["VSKFACT"] = (k, "unstable if std > VSKFACT * VSMEDSTD")
        write_mask(rec_dir / "unstable.fits", unstable, hdr)
        meta["median_pixel_std_adu"] = med_std
        meta["n_unstable"] = int(unstable.sum())
    _finalize(rec_dir, work / "master.fits", kind, meta, frames)
    shutil.rmtree(work, ignore_errors=True)
    return MasterRecord(mid, kind, rec_dir, meta)


def build_flat_master(frames: list[Frame], darkflat: MasterRecord, ctx: BuildContext,
                      sources_root: Path) -> MasterRecord:
    """Master flat: each flat minus the master dark-flat, multiplicative normalisation, rejection."""
    rej = choose_rejection(len(frames), ctx.cfg)
    params = {"rejection": [rej.code, rej.low, rej.high], "norm": "mul", "darkflat": darkflat.id}
    mid = compute_master_id("flat", frames, params)
    found = ctx.library.find("flat", mid)
    if found:
        log.info("Master flat %s déjà en bibliothèque (%s)", mid, found.dir.name)
        return found

    filt = frames[0].filter or "x"
    label = _label("flat", frames, f"_{filt}_{min(f.date_obs for f in frames):%Y%m%d}")
    log.info("Construction du master flat %s (%d images, %s)", label, len(frames), rej.name)
    stats, _ = frame_stats(frames, temporal=False)
    m = ctx.cfg["masters"]
    lo, hi = float(m["flat_level_min_frac"]), float(m["flat_level_max_frac"])
    level_warnings = [s["file"] for s in stats if not lo <= s["frac_saturation"] <= hi]

    work = ctx.tmp_root / f"flat_{mid}"
    if work.exists():
        shutil.rmtree(work)
    link_sequence([f.path for f in frames], work, "f")
    (work / "mdflat.fits").symlink_to(darkflat.path)
    ctx.siril.run(["calibrate f -dark=mdflat -prefix=pp_",
                   f"stack pp_f {rej.siril_args()} -norm=mul -out=master -32b"], work, f"flat_{mid}")

    rec_dir = ctx.library.new_dir("flat", label, mid)
    master_adu = read_siril_output(work / "master.fits")[0]
    meta = {
        "id": mid, "kind": "flat", "label": label, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "siril_version": ctx.siril.version(), "algo_version": MASTER_ALGO_VERSION,
        "rejection": {"code": rej.code, "low": rej.low, "high": rej.high, "name": rej.name},
        "n_frames": len(frames), "filter": frames[0].filter, "exptime": frames[0].exptime,
        "gain": frames[0].gain, "offset": frames[0].offset, "xbin": frames[0].xbin,
        "naxis1": frames[0].naxis1, "naxis2": frames[0].naxis2, "instrument": frames[0].instrument,
        "darkflat_id": darkflat.id, "darkflat_exptime": darkflat.meta.get("exptime"),
        "sources": [str(f.path) for f in frames], "sources_root": str(sources_root),
        "median_adu": float(np.median(master_adu[::4, ::4])),
        "level_min_frac": min(s["frac_saturation"] for s in stats),
        "level_max_frac": max(s["frac_saturation"] for s in stats),
        "level_warnings": level_warnings,
        "frame_stats": stats,
    }
    _finalize(rec_dir, work / "master.fits", "flat", meta, frames)
    shutil.rmtree(work, ignore_errors=True)
    return MasterRecord(mid, "flat", rec_dir, meta)
