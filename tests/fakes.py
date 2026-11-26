"""In-memory fakes of the application ports (tests of the use cases, without files or Siril).

Each fake is checked against the real adapter by the contract tests (tests/contracts/), so the
use-case tests can rely on them.
"""

from __future__ import annotations

import copy
import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable

from varstar.domain.model import Frame, Inventory, SourceScan


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

    def _kept(self, cfg: dict[str, Any], exclude: list[Path] | None) -> list[Frame]:
        patterns = [p.lower() for p in cfg.get("inventory", {}).get("ignore_dirs") or []]
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
        self.writes: list[str] = []

    def exists(self, rel: str) -> bool:
        return rel in self.files

    def read_json(self, rel: str) -> Any | None:
        return json.loads(self.files[rel]) if rel in self.files else None

    def write_text(self, rel: str, text: str) -> None:
        self.files[rel] = text
        self.writes.append(rel)

    def write_json(self, rel: str, data: Any) -> None:
        self.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False))

    def write_yaml(self, rel: str, data: Any) -> None:
        self.write_text(rel, json.dumps(data, default=str))   # the format does not matter to the use cases


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
        return json.dumps({s: cfg.get(s) for s in sections}, sort_keys=True, default=str)

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
