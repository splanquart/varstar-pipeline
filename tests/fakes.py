"""In-memory fakes of the application ports (tests of the use cases, without files or Siril).

Each fake is checked against the real adapter by the contract tests (tests/contracts/), so the
use-case tests can rely on them.
"""

from __future__ import annotations

import copy
import fnmatch
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from varstar.domain.astrometry import CatalogStars, SolveJob, SolveResult, SolveSettings, WcsSolution
from varstar.domain.calibration import CalibrationSensor, FileStat, MasterRecord, PipelineError, Rejection
from varstar.domain.model import Frame, Inventory, SourceScan
from varstar.domain.plan import GroupPlan


class InMemoryFrameSource:
    """``FrameSource`` over frames built in memory (paths are virtual, under ``root``)."""

    def __init__(self, frames: list[Frame], *, unreadable: list[str] | None = None,
                 ignored_dirs: list[str] | None = None, empty_dirs: list[str] | None = None,
                 other_extensions: dict[str, int] | None = None):
        self.frames = frames
        self.unreadable = unreadable or []
        self.ignored_dirs = ignored_dirs or []
        self.empty_dirs = empty_dirs or []
        self.other_extensions = other_extensions or {}
        self.scans = 0
        self.mtimes: dict[str, int] = {}      # path → mtime_ns (default 0): "touch" a file in a test

    def file_stat(self, path) -> FileStat:
        return FileStat(str(path), 1000, self.mtimes.get(str(path), 0))

    def _kept(self, cfg: dict[str, Any], exclude: list[Path] | None) -> list[Frame]:
        patterns = [p.lower() for p in cfg.inventory.ignore_dirs]
        excluded = [str(p) for p in exclude or []]
        kept = []
        for f in self.frames:
            parts = Path(f.rel_path).parent.parts
            if any(fnmatch.fnmatch(part.lower(), p) for part in parts for p in patterns):
                continue
            if any(str(f.path).startswith(e + "/") for e in excluded):
                continue
            kept.append(f)
        return sorted(kept, key=lambda f: f.rel_path)

    def files(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> list[Path]:
        return [f.path for f in self._kept(cfg, exclude)]

    def scan(self, root: Path, cfg: dict[str, Any], exclude: list[Path] | None = None) -> SourceScan:
        self.scans += 1
        return SourceScan(root, copy.deepcopy(self._kept(cfg, exclude)), list(self.unreadable),
                          list(self.ignored_dirs), list(self.empty_dirs), dict(self.other_extensions))


class FakePixelReader:
    """``PixelReader`` returning a fixed scale, or raising for the given relative paths."""

    def __init__(self, scale: int = 4, failing: Iterable[str] = ()):
        self.scale = scale
        self.failing = set(failing)
        self.read: list[str] = []

    def adu_scale(self, frame: Frame) -> int:
        self.read.append(frame.rel_path)
        if frame.rel_path in self.failing:
            raise OSError("pixels illisibles")
        return self.scale


class InMemoryArtifactStore:
    """``ArtifactStore`` keeping the texts in a dict (JSON stored as text, like on disk)."""

    def __init__(self, root: Path = Path("/work")):
        self.root = root
        self.files: dict[str, str] = {}
        self.images: dict[str, np.ndarray] = {}
        self.headers: dict[str, dict[str, Any]] = {}
        self.writes: list[str] = []

    def exists(self, rel: str) -> bool:
        return rel in self.files or rel in self.images

    def read_json(self, rel: str) -> Any | None:
        return json.loads(self.files[rel]) if rel in self.files else None

    def write_text(self, rel: str, text: str) -> None:
        self.files[rel] = text
        self.writes.append(rel)

    def write_json(self, rel: str, data: Any) -> None:
        self.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False, default=str))

    def write_yaml(self, rel: str, data: Any) -> None:
        self.write_text(rel, json.dumps(data, default=str))   # the format does not matter to the use cases

    def write_image(self, rel: str, data: np.ndarray) -> None:
        self.images[rel] = np.asarray(data, dtype=np.float32).copy()
        self.writes.append(rel)

    def write_mask(self, rel: str, mask: np.ndarray, header: dict[str, Any] | None = None) -> None:
        self.images[rel] = np.asarray(mask, dtype=np.uint8).copy()
        self.headers[rel] = dict(header or {})
        self.writes.append(rel)


class FakeProvenance:
    """``ProvenanceRecorder`` with controllable fingerprints; records everything in memory."""

    def __init__(self, files: str = "files-v1", code: str = "code-v1"):
        self.files_value = files
        self.code_value = code
        self.manifests: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def files_fingerprint(self, paths: Iterable[Path]) -> str:
        return f"{self.files_value}:{len(list(paths))}"

    def code_fingerprint(self) -> str:
        return self.code_value

    def config_fingerprint(self, cfg: dict[str, Any], sections: Iterable[str]) -> str:
        return json.dumps({s: cfg.raw.get(s) for s in sections}, sort_keys=True, default=str)

    def read_manifest(self, step: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.manifests.get(step))

    def write_manifest(self, step: str, data: dict[str, Any]) -> None:
        self.manifests[step] = copy.deepcopy(data)

    def record(self, step: str, cfg: dict[str, Any], details: dict[str, Any]) -> None:
        self.records[step] = copy.deepcopy(details)


class RecordingInventoryReport:
    """``InventoryReport`` producing minimal content and remembering what it was given."""

    def __init__(self):
        self.qa_items: list[Any] | None = None

    def to_dict(self, inv: Inventory) -> dict[str, Any]:
        return {"levels": {f"{g.session}/{g.filter}": g.level for g in inv.light_groups},
                "anomalies": [a.code for a in inv.anomalies]}

    def markdown(self, inv: Inventory, qa_items: list[Any]) -> str:
        self.qa_items = qa_items
        return f"# Inventaire\n{len(inv.light_groups)} groupe(s), {len(qa_items)} contrôle(s)\n"


class FakePixelStats:
    """``PixelStats`` answering from tables: statistics per relative path, flats per set id, images per path.

    ``stats`` maps ``frame.rel_path`` to a statistics dict (``builders.make_frame_stats``);
    ``default`` builds the statistics of any other frame (``None``: ``KeyError``, like a missing measure).
    """

    def __init__(self, stats: dict[str, dict[str, Any]] | None = None,
                 default: Callable[[Frame], dict[str, Any]] | None = None,
                 flats: dict[str, np.ndarray] | None = None, images: dict[str, np.ndarray] | None = None,
                 default_image: np.ndarray | None = None):
        self.default_image = default_image
        self.stats = stats or {}
        self.default = default
        self.flats = flats or {}
        self.images = {str(k): v for k, v in (images or {}).items()}
        self.measured: list[str] = []
        self.flat_calls: list[tuple[str, int, int, float]] = []

    def frame_stats(self, frames: list[Frame], saturation: float,
                    progress: str | None = None) -> dict[str, dict[str, Any]]:
        out = {}
        for f in frames:
            self.measured.append(f.rel_path)
            st = self.stats.get(f.rel_path)
            if st is None:
                if self.default is None:
                    raise KeyError(f.rel_path)
                st = self.default(f)
            out[str(f.path)] = copy.deepcopy(st)
        return out

    def quick_flat(self, set_id: str, frames: list[Frame], bin_px: int, dark_level: float) -> np.ndarray:
        self.flat_calls.append((set_id, len(frames), bin_px, dark_level))
        flat = self.flats.get(set_id)
        if flat is None:
            flat = np.ones((8, 8), dtype=np.float32)
        return (flat / np.median(flat)).astype(np.float32)

    def read_image(self, path) -> np.ndarray:
        if str(path) not in self.images:
            if self.default_image is not None:
                return np.asarray(self.default_image, dtype=np.float32).copy()
            raise FileNotFoundError(path)
        return np.asarray(self.images[str(path)], dtype=np.float32).copy()


class RecordingQAReport:
    """``QAReport`` producing a one-line summary and remembering the reports rendered."""

    def __init__(self):
        self.rendered: list[dict[str, Any]] = []

    def markdown(self, report: dict[str, Any]) -> str:
        self.rendered.append(copy.deepcopy(report))
        counts = {s: len(report.get(s, [])) for s in ("header", "raw", "calibrated")}
        return f"# Contrôles qualité\n{counts}\n"


# --------------------------------------------------------------------------- masters and calibration


class InMemoryMasterRepository:
    """``MasterRepository`` keeping masters (record + image) in memory."""

    def __init__(self, root: Path = Path("/lib")):
        self.root = root
        self.records: dict[tuple[str, str], MasterRecord] = {}
        self.images: dict[str, np.ndarray] = {}
        self.unstable: dict[str, np.ndarray] = {}

    def add(self, rec: MasterRecord, image: np.ndarray, unstable: np.ndarray | None = None) -> None:
        self.records[(rec.kind, rec.id)] = rec
        self.images[rec.id] = np.asarray(image, dtype=np.float32)
        if unstable is not None:
            self.unstable[rec.id] = unstable

    def find(self, kind: str, master_id: str) -> MasterRecord | None:
        return self.records.get((kind, master_id))

    def load_adu(self, rec: MasterRecord) -> np.ndarray:
        return self.images[rec.id].copy()

    def load_unstable(self, rec: MasterRecord) -> np.ndarray | None:
        u = self.unstable.get(rec.id)
        return None if u is None else u.copy()


class FakeMasterBuilder:
    """``MasterBuilder`` making constant masters (dark: ``level`` ADU, flat: 1) without Siril."""

    def __init__(self, repo: InMemoryMasterRepository, shape: tuple[int, int] = (64, 96), level: float = 500.0,
                 flat_level_frac: float = 0.5):
        self.repo = repo
        self.shape = shape
        self.level = level
        self.flat_level_frac = flat_level_frac
        self.built: list[tuple[str, str, int]] = []      # (kind, label, n frames)

    def version(self) -> str:
        return "1.4.5"

    def status(self) -> dict[str, Any]:
        return {"mode": "off", "active": False}

    def _meta(self, kind, frames, rejection, master_id, label, sources_root, **extra):
        return {"id": master_id, "kind": kind, "label": label, "rejection": rejection.to_dict(),
                "n_frames": len(frames), "exptime": frames[0].exptime, "sources": [str(f.path) for f in frames],
                "sources_root": str(sources_root), **extra}

    def build_dark(self, frames: list[Frame], kind: str, rejection: Rejection, master_id: str, label: str,
                   sources_root: Path) -> MasterRecord:
        self.built.append((kind, label, len(frames)))
        level = self.level * frames[0].exptime / 300.0 if kind == "dark" else self.level
        stats = [{"file": f.rel_path, "ccd_temp": f.ccd_temp, "median": level, "robust_sigma": 12.0,
                  "corners": {"haut_gauche": 0.0, "haut_droit": 0.0, "bas_gauche": 0.0, "bas_droit": 0.0}}
                 for f in frames]
        meta = self._meta(kind, frames, rejection, master_id, label, sources_root, median_adu=level,
                          sigma_single_frame_adu=12.0 if kind == "dark" else None, frame_stats=stats,
                          **({"n_unstable": 0} if kind == "dark" else {}))
        rec = MasterRecord(master_id, kind, self.repo.root / "darks" / f"{label}_{master_id}", meta)
        image = np.full(self.shape, level, np.float32)
        unstable = np.zeros(self.shape, np.uint8) if kind == "dark" else None
        self.repo.add(rec, image, unstable)
        return rec

    def build_flat(self, frames: list[Frame], darkflat: MasterRecord, rejection: Rejection, master_id: str,
                   label: str, sources_root: Path) -> MasterRecord:
        self.built.append(("flat", label, len(frames)))
        frac = self.flat_level_frac
        meta = self._meta("flat", frames, rejection, master_id, label, sources_root, filter=frames[0].filter,
                          darkflat_id=darkflat.id, median_adu=frac * 65532, level_min_frac=frac,
                          level_max_frac=frac, level_warnings=[] if 0.15 <= frac <= 0.85 else
                          [f.rel_path for f in frames], frame_stats=[])
        rec = MasterRecord(master_id, "flat", self.repo.root / "flats" / f"{label}_{master_id}", meta)
        self.repo.add(rec, np.full(self.shape, frac * 65532, np.float32))
        return rec


class FakeReferenceMasters:
    """``ReferenceMasters`` with images per master kind (``None``: not found; an exception: unreadable)."""

    def __init__(self, images: dict[str, Any]):
        self.images = images

    def find(self, rec: MasterRecord, folder_labels: list[str]) -> str | None:
        return f"/pi/master{rec.kind}_{rec.id}.xisf" if self.images.get(rec.kind) is not None else None

    def load_adu(self, path: str) -> np.ndarray:
        kind = path.split("/master")[1].split("_")[0]
        image = self.images[kind]
        if isinstance(image, Exception):
            raise image
        return np.asarray(image, dtype=np.float32)


class FakeCalibrator:
    """``Calibrator`` writing a placeholder per output into an ``InMemoryArtifactStore``."""

    def __init__(self, store: InMemoryArtifactStore, failing: Iterable[str] = ()):
        self.store = store
        self.failing = set(failing)
        self.calls: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"mode": "off", "active": False}

    def calibrate(self, group: GroupPlan, outputs: dict[str, Path], sensor: CalibrationSensor) -> None:
        self.calls.append(group.key)
        if group.key in self.failing:
            raise RuntimeError("Siril a échoué")
        for out in outputs.values():
            self.store.write_text(str(Path(out).relative_to(self.store.root)), f"calibrated {group.level}")


class RecordingMastersReport:
    def __init__(self):
        self.calls: list[tuple[dict[str, Any], Any, dict[str, Any]]] = []

    def markdown(self, plan: dict[str, Any], qa_items, masks: dict[str, dict[str, Any]]) -> str:
        self.calls.append((copy.deepcopy(plan), qa_items, copy.deepcopy(masks)))
        return f"# Masters\n{len(plan['masters'])} master(s)\n"


class RecordingReporter:
    """``Reporter`` keeping the progress lines."""

    def __init__(self):
        self.lines: list[str] = []

    def progress(self, message: str) -> None:
        self.lines.append(message)


class FakeStarCatalog:
    """``StarCatalog`` answering from fixed stars (filtered like the real adapters), or failing."""

    def __init__(self, stars: CatalogStars | None = None, error: str | None = None):
        self.stars = stars if stars is not None else CatalogStars(np.array([60.44]), np.array([36.11]),
                                                                  np.array([8.0]))
        self.error = error
        self.calls: list[tuple[float, float, float, float]] = []

    def describe(self) -> str:
        return "faux catalogue"

    def cone(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars:
        from varstar.domain.astrometry import angular_sep_arcsec
        self.calls.append((ra, dec, radius_deg, mag_limit))
        if self.error:
            raise PipelineError(self.error)
        s = self.stars
        keep = (angular_sep_arcsec(s.ra, s.dec, ra, dec) / 3600 <= radius_deg) & (s.mag < mag_limit)
        return CatalogStars(s.ra[keep], s.dec[keep], s.mag[keep], self.describe()).sorted_by_mag()


class FakePlateSolver:
    """``PlateSolver`` answering from the hints: a TAN solution centred on the header RA/DEC.

    ``orientation(path)`` gives the orientation of each image (meridian flips); paths in ``failing``
    are not solved.
    """

    def __init__(self, failing: Iterable[str] = (), orientation: Callable[[str], float] | None = None,
                 rms_arcsec: float = 0.3):
        self.failing = set(failing)
        self.orientation = orientation or (lambda path: 0.0)
        self.rms_arcsec = rms_arcsec
        self.calls: list[list[str]] = []

    def name(self) -> str:
        return "faux solveur"

    def solve(self, series: list[list[SolveJob]], catalog: CatalogStars, settings: SolveSettings,
              progress=None) -> dict[str, SolveResult]:
        from builders import make_solution
        out = {}
        for jobs in series:
            self.calls.append([j.path for j in jobs])
            for j in jobs:
                if j.hint is None:
                    out[j.path] = SolveResult(False, reason=j.problem)
                elif any(f in j.path for f in self.failing):
                    out[j.path] = SolveResult(False, reason="aucune correspondance", n_stars=12)
                else:
                    sol = make_solution(j.hint.ra, j.hint.dec, scale_arcsec=j.hint.scale_arcsec, width=j.hint.width,
                                        height=j.hint.height, orientation_deg=self.orientation(j.path))
                    out[j.path] = SolveResult(True, sol, None, 300, catalog.n, 290, self.rms_arcsec)
        return out


class RecordingAstrometryReport:
    def __init__(self):
        self.calls: list[tuple[dict[str, Any], list[Any]]] = []

    def markdown(self, summary: dict[str, Any], qa_items: list[Any]) -> str:
        self.calls.append((copy.deepcopy(summary), list(qa_items)))
        return f"# Astrométrie\n{summary['n_solved']}/{summary['n_images']}\n"
