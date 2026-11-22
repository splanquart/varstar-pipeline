"""Ports: what the use cases need from the outside world (implemented in ``varstar.adapters``).

Plain ``typing.Protocol`` classes: any object with these methods fits (adapters, test fakes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from ..domain.model import Frame, Inventory, SourceScan


class FrameSource(Protocol):
    """Raw frames of a source folder (read-only)."""

    def files(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> list[Path]:
        """FITS files to consider (no file is opened)."""

    def scan(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> SourceScan:
        """Headers of every FITS file, unreadable files and folder facts."""


class PixelReader(Protocol):
    """Pixel-level measurements on raw frames (read-only)."""

    def adu_scale(self, frame: Frame) -> int:
        """Largest power of two dividing every pixel value (×4 for 14-bit data stored on 16 bits)."""


class PixelStats(Protocol):
    """Pixel measurements of the quality checks (raw files read-only; results may be cached)."""

    def frame_stats(self, frames: list[Frame], saturation: float,
                    progress: str | None = None) -> dict[str, dict[str, Any]]:
        """``str(frame.path)`` → ``domain.qa.pixels.frame_quick_stats`` of that frame."""

    def quick_flat(self, set_id: str, frames: list[Frame], bin_px: int, dark_level: float) -> np.ndarray:
        """Binned flats combined by ``domain.qa.pixels.combine_quick_flat`` (median 1)."""

    def read_image(self, path: str | Path) -> np.ndarray:
        """Image of a work file (calibrated frame, mask) in float32; ``FileNotFoundError`` if absent."""


Progress = Callable[[str], None]
"""Receives one-line progress messages for the user (French)."""


class ArtifactStore(Protocol):
    """The work directory. Paths are relative to ``root``."""

    root: Path

    def exists(self, rel: str) -> bool: ...

    def read_json(self, rel: str) -> Any | None: ...

    def write_json(self, rel: str, data: Any) -> None: ...

    def write_text(self, rel: str, text: str) -> None: ...

    def write_yaml(self, rel: str, data: Any) -> None: ...

    def write_image(self, rel: str, data: np.ndarray) -> None:
        """FITS image (float32), e.g. the flat ratios of the P4 check."""


class ProvenanceRecorder(Protocol):
    """Fingerprints for idempotence, step manifests and provenance.json."""

    def files_fingerprint(self, paths: Iterable[Path]) -> str: ...

    def code_fingerprint(self) -> str: ...

    def config_fingerprint(self, cfg: dict[str, Any], sections: Iterable[str]) -> str: ...

    def read_manifest(self, step: str) -> dict[str, Any] | None: ...

    def write_manifest(self, step: str, data: dict[str, Any]) -> None: ...

    def record(self, step: str, cfg: dict[str, Any], details: dict[str, Any]) -> None: ...


class InventoryReport(Protocol):
    """inventory.json content and inventory.md text."""

    def to_dict(self, inv: Inventory) -> dict[str, Any]: ...

    def markdown(self, inv: Inventory, qa_items: list[Any]) -> str: ...


class QAReport(Protocol):
    """qa.md text of a qa.json report."""

    def markdown(self, report: dict[str, Any]) -> str: ...
