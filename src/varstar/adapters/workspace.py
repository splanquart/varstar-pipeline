"""The work directory: every output of the pipeline is written here (never next to the raw files)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class Workspace:
    """``ArtifactStore`` on a local work directory (paths relative to ``root``)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def read_json(self, rel: str) -> Any | None:
        p = self.path(rel)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def write_text(self, rel: str, text: str) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def write_json(self, rel: str, data: Any) -> None:
        self.write_text(rel, json.dumps(data, indent=2, ensure_ascii=False))

    def write_yaml(self, rel: str, data: Any) -> None:
        self.write_text(rel, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
