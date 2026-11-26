"""``ReferenceMasters`` on a folder of PixInsight masters (XISF), for cross-validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.calibration import MasterRecord
from .images import NORM_SCALE, read_xisf


class PixInsightMasters:
    def __init__(self, directory: Path):
        self.dir = Path(directory)

    @classmethod
    def from_config(cls, cfg: dict, source_root: Path) -> "PixInsightMasters | None":
        """``paths.pixinsight_masters``: auto = <source>/output/master ; null = disabled."""
        pi_cfg = cfg["paths"].get("pixinsight_masters", "auto")
        directory = (source_root / "output" / "master") if pi_cfg == "auto" else (Path(pi_cfg) if pi_cfg else None)
        return cls(directory) if directory is not None and directory.exists() else None

    def find(self, rec: MasterRecord, folder_labels: list[str]) -> str | None:
        meta = rec.meta
        if rec.kind in ("dark", "darkflat"):
            pattern = f"masterDark*EXPOSURE-{meta['exptime']:.2f}s*.xisf"
        else:
            pattern = None
            for lab in folder_labels:
                cands = sorted(self.dir.glob(f"masterFlat*FILTER-{meta['filter']}_*SESSION-{lab}.xisf"))
                if cands:
                    pattern = cands[0].name
                    break
            if pattern is None:
                return None
        cands = sorted(self.dir.glob(pattern))
        return str(cands[0]) if cands else None

    def load_adu(self, path: str) -> np.ndarray:
        return read_xisf(Path(path)).astype(np.float32) * np.float32(NORM_SCALE)
