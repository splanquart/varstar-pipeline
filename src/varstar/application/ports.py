"""Ports: what the use cases need from the outside world (implemented in ``varstar.adapters``).

Plain ``typing.Protocol`` classes: any object with these methods fits (adapters, test fakes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from ..domain.calibration import CalibrationSensor, FileStat, MasterRecord, Rejection
from ..domain.model import Frame, Inventory, SourceScan
from ..domain.plan import GroupPlan


class FrameSource(Protocol):
    """Raw frames of a source folder (read-only)."""

    def files(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> list[Path]:
        """FITS files to consider (no file is opened)."""

    def scan(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> SourceScan:
        """Headers of every FITS file, unreadable files and folder facts."""

    def file_stat(self, path: str | Path) -> FileStat:
        """Resolved path, size and modification time of a raw file (identity for the caches)."""


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

    def write_mask(self, rel: str, mask: np.ndarray, header: dict[str, Any] | None = None) -> None:
        """FITS mask (uint8) with optional header keywords."""


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


class MasterRepository(Protocol):
    """Library of masters shared between targets, keyed by master id."""

    root: Path

    def find(self, kind: str, master_id: str) -> MasterRecord | None: ...

    def load_adu(self, rec: MasterRecord) -> np.ndarray:
        """Master image in ADU16 (float32)."""

    def load_unstable(self, rec: MasterRecord) -> np.ndarray | None:
        """Unstable-pixel mask of a light dark (None when absent)."""


class MasterBuilder(Protocol):
    """Builds masters from raw frames and stores them in the library (Siril)."""

    def version(self) -> str: ...

    def status(self) -> dict[str, Any]:
        """State of the tool (e.g. Siril workarounds), for provenance.json."""

    def build_dark(self, frames: list[Frame], kind: str, rejection: Rejection, master_id: str, label: str,
                   sources_root: Path) -> MasterRecord:
        """Master dark / dark of flat: mean with rejection, no normalisation."""

    def build_flat(self, frames: list[Frame], darkflat: MasterRecord, rejection: Rejection, master_id: str,
                   label: str, sources_root: Path) -> MasterRecord:
        """Master flat: each flat minus the dark of flat, multiplicative normalisation, rejection."""


class ReferenceMasters(Protocol):
    """Masters made by another tool (PixInsight), for cross-validation."""

    def find(self, rec: MasterRecord, folder_labels: list[str]) -> str | None: ...

    def load_adu(self, path: str) -> np.ndarray:
        """Reference master in ADU16; ``ValueError`` when it cannot be read."""


class Calibrator(Protocol):
    """Calibrates the lights of a group: (light − dark) / flat, float32, ADU16, provenance header."""

    def calibrate(self, group: GroupPlan, outputs: dict[str, Path], sensor: CalibrationSensor) -> None:
        """Write one calibrated image per raw light (``outputs``: raw path → output path)."""


class MastersReport(Protocol):
    """masters.md text."""

    def markdown(self, plan: dict[str, Any], qa_items: list[Any] | None,
                 masks: dict[str, dict[str, Any]]) -> str:
        """``masks``: mask path → its counts/info (bad pixel map, amp glow)."""
