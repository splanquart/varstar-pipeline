"""``MasterRepository`` on disk: the library of masters shared between targets and years.

Layout:
    <library>/darks/<label>_<id>/master.fits   (+ master.json, unstable.fits for light darks)
    <library>/flats/<label>_<id>/master.fits   (+ master.json)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from ..domain.calibration import MasterRecord
from ..domain.model import Frame
from .images import NORM_SCALE, read_adu


class DiskMasterRepository:
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

    def load_adu(self, rec: MasterRecord) -> np.ndarray:
        return read_adu(rec.path)[0]

    def load_unstable(self, rec: MasterRecord) -> np.ndarray | None:
        return fits.getdata(rec.unstable_path) if rec.unstable_path.exists() else None

    def publish(self, rec_dir: Path, data: np.ndarray, header: fits.Header, meta: dict[str, Any],
                frames: list[Frame]) -> None:
        """Write master.fits (with provenance keywords) and master.json."""
        header = header.copy()
        header["VSKIND"] = (meta["kind"].upper(), "varstar master type")
        header["VSID"] = (meta["id"], "varstar master id")
        header["VSNFRAM"] = (len(frames), "number of source frames")
        header["VSREJ"] = (meta["rejection"]["name"], "rejection algorithm")
        header["VSNORM"] = (NORM_SCALE, "ADU16 = value * VSNORM (Siril normalisation)")
        header["VSSIRIL"] = (meta["siril_version"], "Siril version")
        header["VSDATE"] = (meta["created"], "creation date")
        for f in frames:
            header.add_history(f"src {f.rel_path}"[:72])
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header).writeto(rec_dir / "master.fits",
                                                                                   overwrite=True)
        (rec_dir / "master.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
