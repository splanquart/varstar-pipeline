"""Fixtures for the end-to-end tests on the minisky dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from compare_outputs import diff, normalize, normalize_text
from minisky import CATALOG_NAME, file_state, make_minisky, write_catalog, write_config

GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass
class Minisky:
    base: Path          # temporary directory holding everything below
    root: Path          # raw frames (read-only)
    library: Path
    config: Path
    raw_state: list

    def workdir(self, name: str) -> Path:
        return self.base / name

    def replacements(self) -> dict[str, str]:
        """Absolute paths → placeholders (given and resolved forms: macOS /var → /private/var)."""
        out = {}
        for path, tag in ((self.root, "<RAW>"), (self.base, "<TMP>")):
            out[str(path)] = tag
            out[str(path.resolve())] = tag
        return out


@pytest.fixture(scope="session")
def minisky(tmp_path_factory) -> Minisky:
    base = tmp_path_factory.mktemp("minisky")
    root = make_minisky(base / "raw")
    catalog = write_catalog(base / CATALOG_NAME)
    return Minisky(base, root, base / "lib", write_config(base / "config.yaml", base / "lib", catalog),
                   file_state(root))


class Golden:
    """Compare normalised outputs with tests/e2e/golden/<name>.json (or rewrite them)."""

    def __init__(self, update: bool):
        self.update = update

    def check(self, name: str, data: Any, replacements: dict[str, str] | None = None,
              aliases: dict[str, str] | None = None) -> None:
        actual = normalize(json.loads(json.dumps(data, default=str)), replacements, aliases)
        path = GOLDEN_DIR / f"{name}.json"
        if self.update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(actual, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                            encoding="utf-8")
            return
        if not path.exists():
            pytest.fail(f"référence absente : {path.name} — lancer `pytest --update-golden` puis relire le diff")
        expected = json.loads(path.read_text(encoding="utf-8"))
        problems = diff(expected, actual)
        if problems:
            shown = "\n  ".join(problems[:40])
            more = f"\n  … et {len(problems) - 40} autre(s)" if len(problems) > 40 else ""
            pytest.fail(f"{len(problems)} différence(s) avec la référence {path.name} :\n  {shown}{more}\n"
                        "Si le changement est voulu : `pytest --update-golden` et expliquer pourquoi dans le commit.")

    @staticmethod
    def text(output: str, replacements: dict[str, str]) -> list[str]:
        return normalize_text(output, replacements).splitlines()


@pytest.fixture
def golden(request) -> Golden:
    return Golden(request.config.getoption("--update-golden"))
