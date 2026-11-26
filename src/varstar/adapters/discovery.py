"""Recursive discovery of FITS files under a source folder (folders are never trusted for typing),
and the ``FrameSource`` adapter that reads their headers."""

from __future__ import annotations

import fnmatch
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.calibration import FileStat
from ..domain.model import SourceScan
from ..domain.settings import Settings
from .fits_io import read_frame


@dataclass
class DiscoveryResult:
    fits_files: list[Path] = field(default_factory=list)
    ignored_dirs: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)
    other_extensions: Counter = field(default_factory=Counter)


def _ignore_patterns(cfg: Settings) -> list[str]:
    inv = cfg.inventory
    patterns = list(inv.ignore_dirs)
    if inv.include_removed:
        patterns = [p for p in patterns if p.lower() != "removed"]
    return patterns


def discover(root: Path, cfg: Settings, exclude: list[Path] | None = None) -> DiscoveryResult:
    root = root.resolve()
    exts = {e.lower() for e in cfg.inventory.extensions}
    patterns = _ignore_patterns(cfg)
    excluded = {p.resolve() for p in (exclude or [])}
    result = DiscoveryResult()

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        kept = []
        for d in sorted(dirnames):
            full = current / d
            rel = str(full.relative_to(root))
            if d.startswith(".") or full.resolve() in excluded:
                continue
            if any(fnmatch.fnmatch(d.lower(), p.lower()) for p in patterns):
                result.ignored_dirs.append(rel)
                continue
            kept.append(d)
        dirnames[:] = kept

        visible = [f for f in filenames if not f.startswith(".")]
        if not visible and not kept and current != root:
            result.empty_dirs.append(str(current.relative_to(root)))
        for name in sorted(visible):
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                result.fits_files.append(current / name)
            else:
                result.other_extensions[ext or "(sans extension)"] += 1
    return result


class FilesystemFrameSource:
    """``FrameSource`` on a local folder: discovery + primary header of every FITS file (read-only)."""

    def files(self, root: Path, cfg: Settings, exclude: list[Path] | None = None) -> list[Path]:
        return discover(root, cfg, exclude=exclude).fits_files

    def scan(self, root: Path, cfg: Settings, exclude: list[Path] | None = None) -> SourceScan:
        root = root.resolve()
        disc = discover(root, cfg, exclude=exclude)
        frames, unreadable = [], []
        for path in disc.fits_files:
            try:
                frames.append(read_frame(path, root))
            except Exception as exc:
                unreadable.append(f"{path.relative_to(root)} ({exc.__class__.__name__})")
        return SourceScan(root, frames, unreadable, disc.ignored_dirs, disc.empty_dirs, dict(disc.other_extensions))

    def file_stat(self, path: str | Path) -> FileStat:
        p = Path(path)
        st = p.stat()
        return FileStat(str(p.resolve()), st.st_size, st.st_mtime_ns)
