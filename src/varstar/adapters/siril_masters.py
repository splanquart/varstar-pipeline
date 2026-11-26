"""``MasterBuilder`` with Siril: stacks darks / darks of flat / flats and stores them in the library."""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..domain.calibration import (MASTER_ALGO_VERSION, MasterRecord, Rejection, TemporalNoise, flat_level_warnings,
                                  master_frame_stats, unstable_mask)
from ..domain.model import Frame
from ..domain.qa.pixels import robust_sigma
from .images import read_siril_output, write_mask
from .library import DiskMasterRepository
from .siril import Siril, link_sequence


def read_float(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def frame_stats(frames: list[Frame], temporal: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-frame statistics of the source frames and, if ``temporal``, the per-pixel temporal noise."""
    stats = []
    noise = TemporalNoise() if temporal else None
    for f in frames:
        data = read_float(f.path)
        stats.append(master_frame_stats(f, data))
        if noise is not None:
            noise.add(data)
    return stats, (noise.result() if noise is not None else {})


def _read_produced(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32), hdul[0].header.copy()


class SirilMasterBuilder:
    def __init__(self, siril: Siril, library: DiskMasterRepository, tmp_root: Path, cfg: dict[str, Any]):
        self.siril = siril
        self.library = library
        self.tmp_root = Path(tmp_root)
        self.cfg = cfg

    def version(self) -> str:
        return self.siril.version()

    def status(self) -> dict[str, Any]:
        return self.siril.guard_status()

    def _work(self, name: str) -> Path:
        work = self.tmp_root / name
        if work.exists():
            shutil.rmtree(work)
        return work

    def _common(self, kind: str, label: str, mid: str, rej: Rejection, frames: list[Frame]) -> dict[str, Any]:
        return {"id": mid, "kind": kind, "label": label, "created": dt.datetime.now().isoformat(timespec="seconds"),
                "siril_version": self.siril.version(), "algo_version": MASTER_ALGO_VERSION,
                "rejection": rej.to_dict(), "n_frames": len(frames)}

    def build_dark(self, frames: list[Frame], kind: str, rejection: Rejection, master_id: str, label: str,
                   sources_root: Path) -> MasterRecord:
        work = self._work(f"{kind}_{master_id}")
        link_sequence([f.path for f in frames], work, "f")
        self.siril.run([f"stack f {rejection.siril_args()} -nonorm -out=master -32b"], work, f"{kind}_{master_id}")

        stats, extra = frame_stats(frames, temporal=(kind == "dark"))
        rec_dir = self.library.new_dir(kind, label, master_id)
        master_adu = read_siril_output(work / "master.fits")[0]
        temps = [f.ccd_temp for f in frames if f.ccd_temp is not None]
        f0 = frames[0]
        meta = {
            **self._common(kind, label, master_id, rejection, frames),
            "exptime": f0.exptime, "gain": f0.gain, "offset": f0.offset, "xbin": f0.xbin, "ybin": f0.ybin,
            "naxis1": f0.naxis1, "naxis2": f0.naxis2, "instrument": f0.instrument,
            "mean_ccd_temp": float(np.mean(temps)) if temps else None,
            "sources": [str(f.path) for f in frames], "sources_root": str(sources_root),
            "median_adu": float(np.median(master_adu[::4, ::4])),
            "robust_sigma_adu": robust_sigma(master_adu[::4, ::4]),
            "sigma_single_frame_adu": extra.get("sigma_single"),
            "frame_stats": stats,
        }
        if "pixel_std" in extra:
            k = float(self.cfg["masks"].get("unstable_sigma_factor", 5.0))
            unstable, med_std = unstable_mask(extra["pixel_std"], k)
            hdr = fits.Header()
            hdr["VSID"] = master_id
            hdr["VSMEDSTD"] = (med_std, "median per-pixel temporal std (ADU16)")
            hdr["VSKFACT"] = (k, "unstable if std > VSKFACT * VSMEDSTD")
            write_mask(rec_dir / "unstable.fits", unstable, hdr)
            meta["median_pixel_std_adu"] = med_std
            meta["n_unstable"] = int(unstable.sum())
        data, header = _read_produced(work / "master.fits")
        self.library.publish(rec_dir, data, header, meta, frames)
        shutil.rmtree(work, ignore_errors=True)
        return MasterRecord(master_id, kind, rec_dir, meta)

    def build_flat(self, frames: list[Frame], darkflat: MasterRecord, rejection: Rejection, master_id: str,
                   label: str, sources_root: Path) -> MasterRecord:
        stats, _ = frame_stats(frames, temporal=False)
        work = self._work(f"flat_{master_id}")
        link_sequence([f.path for f in frames], work, "f")
        (work / "mdflat.fits").symlink_to(darkflat.path)
        self.siril.run(["calibrate f -dark=mdflat -prefix=pp_",
                        f"stack pp_f {rejection.siril_args()} -norm=mul -out=master -32b"], work, f"flat_{master_id}")

        rec_dir = self.library.new_dir("flat", label, master_id)
        master_adu = read_siril_output(work / "master.fits")[0]
        f0 = frames[0]
        meta = {
            **self._common("flat", label, master_id, rejection, frames),
            "filter": f0.filter, "exptime": f0.exptime, "gain": f0.gain, "offset": f0.offset, "xbin": f0.xbin,
            "naxis1": f0.naxis1, "naxis2": f0.naxis2, "instrument": f0.instrument,
            "darkflat_id": darkflat.id, "darkflat_exptime": darkflat.meta.get("exptime"),
            "sources": [str(f.path) for f in frames], "sources_root": str(sources_root),
            "median_adu": float(np.median(master_adu[::4, ::4])),
            "level_min_frac": min(s["frac_saturation"] for s in stats),
            "level_max_frac": max(s["frac_saturation"] for s in stats),
            "level_warnings": flat_level_warnings(stats, self.cfg),
            "frame_stats": stats,
        }
        data, header = _read_produced(work / "master.fits")
        self.library.publish(rec_dir, data, header, meta, frames)
        shutil.rmtree(work, ignore_errors=True)
        return MasterRecord(master_id, "flat", rec_dir, meta)
