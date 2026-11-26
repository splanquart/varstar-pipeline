"""``ProvenanceRecorder`` on the work directory (manifests/, provenance.json).

Wraps the functions of :mod:`varstar.provenance` until they move here (step 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import provenance as _prov


class ProvenanceStore:
    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)

    def files_fingerprint(self, paths: Iterable[Path]) -> str:
        return _prov.files_fingerprint(paths)

    def code_fingerprint(self) -> str:
        return _prov.code_fingerprint()

    def config_fingerprint(self, cfg: dict[str, Any], sections: Iterable[str]) -> str:
        return _prov.config_fingerprint(cfg, sections)

    def read_manifest(self, step: str) -> dict[str, Any] | None:
        return _prov.read_manifest(self.workdir, step)

    def write_manifest(self, step: str, data: dict[str, Any]) -> None:
        _prov.write_manifest(self.workdir, step, data)

    def record(self, step: str, cfg: dict[str, Any], details: dict[str, Any]) -> None:
        _prov.update_provenance(self.workdir, step, cfg, details)
