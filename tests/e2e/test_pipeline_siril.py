"""End-to-end masters + calibration through the real siril-cli on tiny synthetic frames."""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from varstar.adapters.config_loader import load_settings
from varstar.cli.bootstrap import run_pipeline

from fitsgen import series

pytestmark = pytest.mark.siril

SHAPE = (60, 80)
GRAD = np.linspace(0.8, 1.2, SHAPE[1])[None, :] * np.ones(SHAPE)


def q4(x):
    return (np.round(np.clip(x, 0, 65532) / 4) * 4).astype(np.uint16)


def make_dataset(root: Path, dark_exp=300.0, dark_current=0.0, n_dark=8):
    rng = np.random.default_rng(7)

    def dark(exp):
        def mk(i):
            d = 500 + dark_current * exp + rng.normal(0, 2, SHAPE)
            d[5, 7] += 10 * exp      # hot pixel: 10 ADU/s
            return q4(d)
        return mk

    def flat(i):
        d = 500 + 1.13 * dark_current + 30000 * GRAD + rng.normal(0, 20, SHAPE)
        d[5, 7] += 10 * 1.13
        return q4(d)

    def light(i):
        d = 500 + 300 * dark_current + 1000 * GRAD + rng.normal(0, 3, SHAPE)
        d[5, 7] += 3000
        return q4(d)

    series(root, "light/H/SESSION_s1", "LIGHT", 300, "2021-09-06T23:00:00", 3, step_s=400, filt="H",
           shape=SHAPE, make=light)
    series(root, "flat/SESSION_s1", "FLAT", 1.13, "2021-09-07T06:00:00", 5, step_s=2, filt="H",
           shape=SHAPE, make=flat)
    series(root, "dark/darkflat", "DARKFLAT", 1.13, "2021-09-07T07:00:00", 5, step_s=2, prefix="",
           shape=SHAPE, make=dark(1.13))
    series(root, "dark/darklight", "DARK", dark_exp, "2021-09-07T02:00:00", n_dark, step_s=dark_exp + 10,
           prefix="", shape=SHAPE, make=dark(dark_exp))


def cfg_for(tmp_path):
    return load_settings(overrides={"paths": {"library": str(tmp_path / "lib")}})


def test_end_to_end_level_a(tmp_path):
    root = tmp_path / "raw"
    make_dataset(root)
    raw_state = sorted((p, p.stat().st_mtime_ns) for p in root.rglob("*"))
    wd = tmp_path / "work"
    cfg = cfg_for(tmp_path)
    summary = run_pipeline(root, cfg, wd)

    plan = summary["plan"]
    (key,) = plan["groups"]
    group = plan["groups"][key]
    assert group["level"] == "A" and group["dark"]["mode"] == "direct"
    kinds = sorted(m["kind"] for m in plan["masters"].values())
    assert kinds == ["dark", "darkflat", "flat"]
    dark_master = next(m for m in plan["masters"].values() if m["kind"] == "dark")
    assert dark_master["rejection"] == "sigma clipping"           # N = 8
    assert abs(dark_master["median_adu"] - 500) < 2

    outputs = summary["calibrate"]["results"][key]["outputs"]
    assert len(outputs) == 3
    out = Path(next(iter(outputs.values())))
    assert out.parent.parts[-2:] == ("2021-09-06_s1", "H") and " " not in out.name
    with fits.open(out) as hdul:
        data, h = hdul[0].data, hdul[0].header
    assert data.dtype.kind == "f" and data.dtype.itemsize == 4
    # (light - dark) / normalised flat: the flat gradient is removed, sky level ~1000 ADU.
    assert abs(np.median(data) - 1000) < 15
    cols = np.median(data, axis=0)
    assert cols.max() - cols.min() < 20
    assert abs(data[5, 7] - np.median(data)) < 50            # hot pixel removed by the dark
    assert h["BUNIT"] == "ADU" and h["VSLEVEL"] == "A" and h["DATE-OBS"].startswith("2021-09-06")
    assert h["VSADUSC"] == 4 and abs(h["EGAINEFF"] - 0.25056) < 1e-4
    assert Path(h["VSRAW"]).exists()

    bpm = fits.getdata(group["bpm"])
    assert bpm[5, 7] & 2  # very hot
    assert (wd / "masters" / "masters.md").exists()
    prov = json.loads((wd / "provenance.json").read_text())
    assert {"inventory", "masters", "calibrate"} <= set(prov["steps"])
    assert len(prov["steps"]["calibrate"]["images"]) == 3
    # Raw files untouched.
    assert sorted((p, p.stat().st_mtime_ns) for p in root.rglob("*")) == raw_state

    # Second run: masters from the library, calibration skipped.
    again = run_pipeline(root, cfg, wd)
    assert again["calibrate"]["results"][key]["skipped"]
    assert len(list((tmp_path / "lib").rglob("master.fits"))) == 3
    # --from calibrate reuses the plan.
    again = run_pipeline(root, cfg, wd, from_step="calibrate")
    assert again["steps"] == ["calibrate", "astrometry"]
    # These frames have no RA/DEC: no blind solving, every image is reported as not solved (A1).
    assert again["astrometry"]["n_solved"] == 0
    assert [i["check"] for i in again["astrometry"]["qa"]] == ["A1"]


def test_scaled_dark_level_b(tmp_path):
    root = tmp_path / "raw"
    make_dataset(root, dark_exp=100.0, dark_current=0.5)
    wd = tmp_path / "work"
    summary = run_pipeline(root, cfg_for(tmp_path), wd)
    (group,) = summary["plan"]["groups"].values()
    assert group["level"] == "B" and group["dark"]["mode"] == "scaled"
    assert abs(group["dark"]["offset_adu"] - 500.6) < 3
    out = Path(next(iter(summary["calibrate"]["results"][group["key"]]["outputs"].values())))
    data = fits.getdata(out)
    # light - offset - 3 x (dark100 - offset) = 1000 x flat gradient, flat-fielded -> ~1000.
    assert abs(np.median(data) - 1000) < 20


def test_session_selection_and_errors(tmp_path):
    from varstar.domain.calibration import PipelineError
    root = tmp_path / "raw"
    make_dataset(root)
    with pytest.raises(PipelineError):
        run_pipeline(root, cfg_for(tmp_path), tmp_path / "w", sessions=["nope"])
    with pytest.raises(PipelineError):
        run_pipeline(root, cfg_for(tmp_path), tmp_path / "w2", from_step="calibrate")
