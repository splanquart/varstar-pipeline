"""``PixelStats`` adapter: pixel measurements of the quality checks on FITS files.

Raw files are opened read-only. Results are cached in the work directory:
- ``qa/frame_stats.json``: quick statistics per raw file (keyed by size + mtime);
- ``qa/flats/*.npy``: binned, normalised flats of the P4 comparison.
Several processes are used for large batches.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from astropy.io import fits

from ..domain.model import Frame
from ..domain.qa.pixels import bin_frame, combine_quick_flat, frame_quick_stats
from ..domain.settings import Settings

STATS_VERSION = 3


def read_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw integer data as int64-able array, float32 ADU data) without any BZERO memmap issue."""
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
        hdu = hdul[0]
        raw = np.asarray(hdu.data)
        bzero = hdu.header.get("BZERO", 0)
        bscale = hdu.header.get("BSCALE", 1)
    if bscale == 1 and float(bzero) == int(bzero) and np.issubdtype(raw.dtype, np.integer):
        ints = raw.astype(np.int64) + int(bzero)
        return ints, ints.astype(np.float32)
    data = (raw * bscale + bzero).astype(np.float32)
    return data, data


def quick_stats(path: str, step: int = 4, saturation: float = 65532.0) -> dict[str, Any]:
    """Sub-sampled statistics of one raw file (top-level: runs in worker processes)."""
    ints, data = read_raw(path)
    return frame_quick_stats(ints, data, step, saturation)


def binned_frame(path: str, bin_px: int = 8) -> np.ndarray:
    """Full-resolution frame averaged in bin_px x bin_px blocks (for flat comparisons)."""
    return bin_frame(read_raw(path)[1], bin_px)


def read_image(path: str | Path) -> np.ndarray:
    """Primary image of a FITS file (calibrated frame, mask) in float32."""
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def file_id(f: Frame) -> str:
    st = f.path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


class PixelStatsCache:
    """``PixelStats`` on FITS files, cached under ``<workdir>/qa``."""

    def __init__(self, workdir: Path, *, step: int = 4, processes: int = 1,
                 say: Callable[[str], None] | None = None):
        self.dir = Path(workdir) / "qa"
        self.path = self.dir / "frame_stats.json"
        self.step = int(step)
        self.processes = max(1, int(processes))
        self.say = say or (lambda message: None)
        self._data: dict[str, Any] | None = None

    @property
    def data(self) -> dict[str, Any]:
        """Cached statistics per file path (read from disk on first use)."""
        if self._data is None:
            self._data = {}
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    if raw.get("version") == STATS_VERSION and raw.get("step") == self.step:
                        self._data = raw["files"]
                except (json.JSONDecodeError, KeyError):
                    self._data = {}
        return self._data

    @classmethod
    def from_config(cls, workdir: Path, cfg: Settings,
                    say: Callable[[str], None] | None = None) -> "PixelStatsCache":
        return cls(workdir, step=cfg.qa.pixel_step, processes=cfg.runtime.processes, say=say)

    def frame_stats(self, frames: list[Frame], saturation: float,
                    progress: str | None = None) -> dict[str, dict[str, Any]]:
        todo = [f for f in frames if self.data.get(str(f.path), {}).get("id") != file_id(f)]
        if todo:
            if progress:
                self.say(f"  {progress} : statistiques rapides de {len(todo)} image(s) brute(s) …")
            paths = [str(f.path) for f in todo]
            if self.processes > 1 and len(todo) > 4:
                with ProcessPoolExecutor(self.processes) as ex:
                    results = list(ex.map(quick_stats, paths, [self.step] * len(paths),
                                          [saturation] * len(paths), chunksize=4))
            else:
                results = [quick_stats(p, self.step, saturation) for p in paths]
            for f, r in zip(todo, results):
                self.data[str(f.path)] = {"id": file_id(f), **r}
            self.save()
        return {str(f.path): self.data[str(f.path)] for f in frames}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": STATS_VERSION, "step": self.step, "files": self.data}),
                       encoding="utf-8")
        tmp.replace(self.path)

    def quick_flat(self, set_id: str, frames: list[Frame], bin_px: int, dark_level: float) -> np.ndarray:
        key = f"{set_id}_{bin_px}_{len(frames)}_{dark_level:.1f}_{file_id(frames[0])}"
        cache = self.dir / "flats" / f"{key}.npy"
        if cache.exists():
            return np.load(cache)
        paths = [str(f.path) for f in frames]
        if self.processes > 1 and len(paths) > 2:
            with ProcessPoolExecutor(min(self.processes, len(paths))) as ex:
                stack = list(ex.map(binned_frame, paths, [bin_px] * len(paths)))
        else:
            stack = [binned_frame(p, bin_px) for p in paths]
        flat = combine_quick_flat(stack, dark_level)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, flat)
        return flat

    def read_image(self, path: str | Path) -> np.ndarray:
        return read_image(path)
