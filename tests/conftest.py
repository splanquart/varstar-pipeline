"""Synthetic N.I.N.A.-like FITS files for unit tests (no dependency on the real dataset)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from hypothesis import settings

from varstar.config import load_config, load_defaults

from fitsgen import series


# Property-based tests: deterministic by default (reproducibility), wide exploration on demand
# with `pytest --hypothesis-profile=explore tests/domain`.
settings.register_profile("default", derandomize=True, max_examples=50, deadline=None)
settings.register_profile("explore", max_examples=1000, deadline=None)
settings.load_profile("default")

SIRIL_CLI = Path(load_defaults()["paths"]["siril_cli"])


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="réécrit les sorties de référence (tests/e2e/golden) au lieu de les comparer")


def pytest_collection_modifyitems(config, items):
    if SIRIL_CLI.exists():
        return
    skip = pytest.mark.skip(reason=f"siril-cli absent ({SIRIL_CLI})")
    for item in items:
        if "siril" in item.keywords:
            item.add_marker(skip)


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


@pytest.fixture
def snapshot(request):
    """``snapshot(name, text)``: compare a rendered text with tests/snapshots/<name> (or rewrite it with
    ``--update-golden``)."""
    import difflib

    update = request.config.getoption("--update-golden")

    def check(name: str, text: str) -> None:
        path = SNAPSHOT_DIR / name
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return
        if not path.exists():
            pytest.fail(f"instantané absent : {name} — lancer `pytest --update-golden` puis relire le diff")
        expected = path.read_text(encoding="utf-8")
        if text != expected:
            delta = "".join(difflib.unified_diff(expected.splitlines(True), text.splitlines(True), name, "obtenu"))
            pytest.fail(f"rendu différent de l'instantané {name} :\n{delta}\n"
                        "Si le changement est voulu : `pytest --update-golden` et expliquer pourquoi dans le commit.")

    return check


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def complete_session(tmp_path: Path) -> Path:
    """One complete night (sept06-like): H/S lights, flats next morning, darks, dark-flats."""
    root = tmp_path / "YEAR"
    series(root, "light/H/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:04:00", 3, step_s=600, filt="H")
    series(root, "light/S/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:09:00", 3, step_s=600, filt="S")
    series(root, "flat/SESSION_sept06", "FLAT", 1.13, "2021-09-07T13:21:00", 3, step_s=2, filt="H")
    series(root, "flat/SESSION_sept06", "FLAT", 2.62, "2021-09-07T13:22:00", 3, step_s=2, filt="S")
    series(root, "dark/darklight/300s", "DARK", 300, "2021-09-07T02:00:00", 4, step_s=310, prefix="")
    series(root, "dark/darkflat", "DARKFLAT", 1.13, "2021-09-08T18:25:00", 3, step_s=2, prefix="")
    series(root, "dark/darkflat", "DARKFLAT", 2.62, "2021-09-07T13:28:00", 3, step_s=3, prefix="")
    return root
