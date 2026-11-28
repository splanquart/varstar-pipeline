"""Integration test on the reference dataset (pytest -m real_data)."""

from pathlib import Path

import pytest

from varstar.adapters.config_loader import load_settings
from varstar.cli.bootstrap import load_inventory as build_inventory
from varstar.domain.inventory import darks_todo

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

DATA = Path("/Users/steph/Pictures/pixinsight/NGC1499/YEAR_2021")

pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not DATA.exists(), reason="jeu de référence absent")]


def test_reference_inventory():
    inv = build_inventory(DATA, load_settings(CONFIG))
    c = inv.counts_by_type()
    assert c["LIGHT"] == 170
    lv = {(inv.session_label(s).split(" ")[1].strip("()"), f): v
          for (s, f), v in inv.levels_by_session_filter().items()}
    assert lv[("aout27", "H")] == "B"
    assert lv[("sept02", "H")] == lv[("sept02", "S")] == "B"  # D11: sept06 flats
    assert lv[("sept06", "H")] == lv[("sept06", "S")] == "A"
    # D16: only aout31, sept02 and sept06 have darks taken the same night; the others borrow → B.
    assert {k for k, v in lv.items() if v == "A"} == {("aout31", "H"), ("aout31", "S"), ("sept06", "H"),
                                                       ("sept06", "S")}
    assert [(t.kind, t.exptime) for t in darks_todo(inv)] == [("darkflat", 1.61)]
    assert inv.sensors[0].adu_scale == 4
    assert any(a.code == "folder_exptime" and len(a.files) == 2 for a in inv.anomalies)


WORKDIR = DATA.parent / "varstar_work" / DATA.name


@pytest.mark.skipif(not (WORKDIR / "manifests" / "calibrate").exists(), reason="sept06 pas encore calibrée")
def test_reference_astrometry_sept06():
    """M3 acceptance: every sept06 image solved against Gaia DR3 (cached after the first run)."""
    from varstar.cli.bootstrap import run_pipeline
    cfg = load_settings(CONFIG)
    summary = run_pipeline(DATA, cfg, WORKDIR, from_step="astrometry", sessions=["sept06"])
    ast = summary["astrometry"]
    assert ast["n_images"] == 29 and ast["n_solved"] == 29
    for row in ast["rows"]:
        assert row["rms_median_arcsec"] < 0.6
        assert abs(row["scale_median_arcsec"] - 3.486) < 0.005        # SPEC §4.4: ≈ 3.49″/px
        assert row["segments"] == [1]                                  # no meridian flip that night
