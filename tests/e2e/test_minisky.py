"""End-to-end tests through the CLI on the minisky dataset (tests/e2e/minisky.py).

Two kinds of assertions:
- explicit ones, one per planted defect (readable, they document the expected behaviour);
- golden comparisons of the normalised outputs (tests/e2e/golden/), which catch any other
  change of behaviour during the refactoring. Update them on purpose only:
  `pytest --update-golden`, and explain why in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from typer.testing import CliRunner

import minisky as M
from compare_outputs import image_signature, master_aliases
from varstar.cli import app
from varstar.domain.masks import BIT_NAMES

runner = CliRunner()
BIT = {name: bit for bit, name in BIT_NAMES.items()}
EXPECTED_LEVELS = {("s1", "H"): "A", ("s1", "S"): "A", ("s2", "H"): "B", ("s3", "S"): "C"}
FLAT_FIELDED = {"s1", "s2"}


def invoke(*args: str):
    res = runner.invoke(app, [str(a) for a in args])
    assert res.exception is None or isinstance(res.exception, SystemExit), res.output
    return res


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_median(img: np.ndarray, yx: tuple[int, int], r: int = 3) -> float:
    y, x = yx
    win = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].astype(float).copy()
    win[min(y, r), min(x, r)] = np.nan
    return float(np.nanmedian(win))


def star_flux_ratios(img: np.ndarray, curved: bool = False) -> list[float]:
    """Aperture flux / injected flux of each planted star (background from an annulus r 6–9 px).

    ``curved``: quadratic background fitted on the annulus — the median is biased by the curvature
    of a vignetted sky (level C, no flat); on a flat background the median is less noisy.
    """
    y, x = np.indices(img.shape)
    out = []
    for (yc, xc), flux in M.STARS:
        r = np.hypot(y - yc, x - xc)
        ring, ap = (r > 6) & (r < 9), r < 5
        dy, dx = y - yc, x - xc
        basis = lambda m: np.column_stack([np.ones(m.sum()), dx[m], dy[m], dx[m] ** 2, dy[m] ** 2])  # noqa: E731
        if curved:
            coef, *_ = np.linalg.lstsq(basis(ring), img[ring].astype(float), rcond=None)
            out.append(float((img[ap] - basis(ap) @ coef).sum() / flux))
        else:
            out.append(float((img[ap] - np.median(img[ring])).sum() / flux))
    return out


# =========================================================================== without Siril


@pytest.fixture(scope="module")
def inventory_run(minisky):
    wd = minisky.workdir("work_inventory")
    res = invoke("inventory", minisky.root, "-c", minisky.config, "-w", wd)
    return res, wd


@pytest.fixture(scope="module")
def check_run(minisky):
    wd = minisky.workdir("work_check")
    res = invoke("check", minisky.root, "-c", minisky.config, "-w", wd)
    return res, wd


def test_inventory_levels(inventory_run):
    res, wd = inventory_run
    assert res.exit_code == 0, res.output
    inv = load(wd / "inventory.json")
    levels = {(g["label"], g["filter"]): g["level"] for g in inv["light_groups"]}
    assert levels == EXPECTED_LEVELS
    # No silent degradation: every component below A says why.
    for g in inv["light_groups"]:
        for part in ("dark", "flat", "darkflat"):
            m = g[part]
            if m and m["level"] != "A":
                assert m["reasons"], (g["label"], g["filter"], part)


def test_inventory_traps(inventory_run):
    _, wd = inventory_run
    inv = load(wd / "inventory.json")
    by_code = {a["code"]: a for a in inv["anomalies"]}
    # A flat misfiled in a dark folder is classified by its header, and reported.
    assert by_code["folder_type"]["files"] == ["dark/oops/2021-09-07_06-00-08_NGC 1499_H_1.13s_120_0000.fits"]
    # The s2 darks taken at −5 °C are not used (ΔT > tolerance): s2 borrows the s1 darks (level B).
    assert "DARK-300s-20210910" in by_code["unused_sets"]["message"]
    s2 = next(g for g in inv["light_groups"] if g["label"] == "s2")
    assert s2["dark"]["sets"] == ["DARK-300s-20210907"]
    # s3 has no flat within 7 days.
    s3 = next(g for g in inv["light_groups"] if g["label"] == "s3")
    assert s3["flat"]["level"] == "C" and not s3["flat"]["sets"]
    # The ×4 ADU scale is detected from the pixels, not hard-coded.
    (sensor,) = inv["sensors"]
    assert sensor["adu_scale"] == 4 and abs(sensor["effective_gain_e_per_adu"] - 0.25056) < 1e-4
    md = (wd / "inventory.md").read_text(encoding="utf-8")
    assert "s3" in md and "Niveaux de calibration" in md


def test_check_command(check_run, minisky):
    res, wd = check_run
    assert res.exit_code == 0, res.output
    qa = load(wd / "qa" / "qa.json")
    items = qa["header"] + qa["raw"]
    checks = {(i["check"], i["severity"]) for i in items}
    assert ("H1", "warning") in checks          # misfiled flat
    # The s2 darks (sensor at −5 °C instead of −12 °C) are a temperature change, not a light leak:
    # https://github.com/splanquart/varstar-pipeline/issues/3
    warm = [i for i in items if i["check"] in ("P1", "P2", "P9") and i["severity"] == "warning"
            and any("2021-09-10" in f for f in i["files"])]
    assert [i["check"] for i in warm] == ["P9"] and warm[0]["data"]["delta_temp_c"] == 7.0
    assert (wd / "qa" / "qa.md").exists()
    assert M.file_state(minisky.root) == minisky.raw_state


def test_darks_todo_command(minisky):
    res = invoke("darks-todo", minisky.root, "-c", minisky.config)
    assert res.exit_code == 0, res.output
    assert "Aucun dark manquant" in res.output


def test_golden_inventory(inventory_run, minisky, golden):
    res, wd = inventory_run
    repl = minisky.replacements()
    golden.check("inventory", {"exit_code": res.exit_code, "stdout": golden.text(res.output, repl),
                               "inventory.json": load(wd / "inventory.json"),
                               "inventory.md": golden.text((wd / "inventory.md").read_text(encoding="utf-8"), repl)},
                 repl)


def test_golden_check(check_run, minisky, golden):
    res, wd = check_run
    repl = minisky.replacements()
    golden.check("check", {"exit_code": res.exit_code, "stdout": golden.text(res.output, repl),
                           "qa.json": load(wd / "qa" / "qa.json"),
                           "qa.md": golden.text((wd / "qa" / "qa.md").read_text(encoding="utf-8"), repl)},
                 repl)


def test_golden_darks_todo(minisky, golden):
    res = invoke("darks-todo", minisky.root, "-c", minisky.config)
    repl = minisky.replacements()
    golden.check("darks_todo", {"exit_code": res.exit_code, "stdout": golden.text(res.output, repl)}, repl)


# =========================================================================== with Siril


@pytest.fixture(scope="module")
def siril_run(minisky):
    wd = minisky.workdir("work_run")
    res = invoke("run", minisky.root, "-c", minisky.config, "-w", wd)
    plan = load(wd / "masters" / "plan.json")
    groups = {(g["label"], g["filter"]): g for g in plan["groups"].values()}
    return res, wd, plan, groups


def calibrated(wd: Path, group: dict) -> list[Path]:
    prov = load(wd / "provenance.json")
    images = prov["steps"]["calibrate"]["images"]
    return [Path(images[f]) for f in group["files"]]


@pytest.mark.siril
def test_run_levels_and_masters(siril_run):
    res, _, plan, groups = siril_run
    assert res.exit_code == 0, res.output
    assert {k: g["level"] for k, g in groups.items()} == EXPECTED_LEVELS
    assert all(g["dark"] and g["dark"]["mode"] == "direct" for g in groups.values())
    assert groups[("s3", "S")]["flat"] is None
    assert all(groups[k]["flat"] for k in groups if k[0] in FLAT_FIELDED)
    kinds = sorted(m["kind"] for m in plan["masters"].values())
    assert kinds == ["dark", "dark", "darkflat", "darkflat", "flat", "flat", "flat"]


@pytest.mark.siril
def test_run_bad_pixel_mask(siril_run):
    _, _, _, groups = siril_run
    for (label, _), g in groups.items():
        bpm = fits.getdata(g["bpm"])
        assert bpm[M.HOT] & BIT["very_hot"], label
        assert bpm[M.COSMIC] == 0, label               # a light-only event is not a sensor defect
        if label in FLAT_FIELDED:
            assert bpm[M.DEAD] & BIT["dead"], label
        else:
            assert bpm[M.DEAD] & BIT["dead"] == 0, label   # no flat → low response cannot be seen


@pytest.mark.siril
def test_run_amp_glow_mask(siril_run):
    _, _, _, groups = siril_run
    for g in groups.values():
        mask = fits.getdata(g["ampglow"])
        assert mask[M.AMP_GLOW].all()
        outside = mask.copy()
        outside[M.AMP_GLOW] = 0
        assert not outside.any()


@pytest.mark.siril
def test_run_calibrated_images(siril_run):
    _, wd, _, groups = siril_run
    for (label, _), g in groups.items():
        outs = calibrated(wd, g)
        assert len(outs) == 3
        for i, out in enumerate(outs):
            assert " " not in out.name
            with fits.open(out) as hdul:
                data, h = hdul[0].data, hdul[0].header
            assert h["BITPIX"] == -32 and data.dtype.kind == "f" and data.dtype.itemsize == 4
            assert h["VSLEVEL"] == g["level"] and h["BUNIT"] == "ADU" and h["VSADUSC"] == 4
            # Hot pixel removed by the dark, but no cosmetic correction: the cosmic ray of the
            # first s1/H light is still there.
            assert abs(data[M.HOT] - local_median(data, M.HOT)) < 30
            excess = data[M.COSMIC] - local_median(data, M.COSMIC)
            if (label, g["filter"], i) == ("s1", "H", 0):
                assert excess > 0.75 * M.COSMIC_ADU
            else:
                assert abs(excess) < 30
            ratios = star_flux_ratios(data, curved=label not in FLAT_FIELDED)
            if label in FLAT_FIELDED:
                # Vignetting corrected: flat background, same star flux everywhere in the field.
                cols = np.median(data, axis=0)
                assert cols.max() - cols.min() < 15
                # The three bright stars of M2 within 2 %; the fainter M3 stars (photon noise ≈ 1.7 %) within
                # 5 % — without the flat, the vignetting alone spreads them by 15 %.
                assert max(ratios[:3]) - min(ratios[:3]) < 0.02
                assert max(ratios) - min(ratios) < 0.05
                assert abs(np.median(data[M.AMP_GLOW]) - np.median(data[16:32, 80:96])) < 5
            else:
                # Level C (no flat): the vignetting is left in the image, star by star.
                # Expected: the vignetting averaged over each star's profile (steep near the edges).
                yy, xx = np.indices(M.SHAPE)
                vig = M.vignetting()
                expected = []
                for (yc, xc), _ in M.STARS:
                    psf = np.exp(-((yy - yc) ** 2 + (xx - xc) ** 2) / (2 * M.STAR_SIGMA ** 2))
                    expected.append(float((psf * vig).sum() / psf.sum()))
                assert np.allclose(ratios, expected, atol=0.02), (ratios, expected)


@pytest.mark.siril
def test_run_astrometry(siril_run):
    """M3: every light is solved against the minisky catalogue; the WCS is the planted one."""
    from varstar.domain.astrometry import WcsSolution
    res, wd, _, groups = siril_run
    assert "Astrométrie : 12/12 image(s) résolue(s)" in res.output
    truth = M.true_wcs()
    h, w = M.SHAPE
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]], float)
    ra, dec = truth.pixel_to_world(corners[:, 0], corners[:, 1])
    for g in groups.values():
        sol = load(wd / "astrometry" / f"{g['key']}.json")
        assert len(sol["images"]) == 3
        for raw, row in sol["images"].items():
            assert row["ok"], row["reason"]
            assert row["n_matched"] == len(M.STARS) and row["rms_arcsec"] < 0.5
            wcs = WcsSolution.from_dict(row["wcs"])
            x, y = wcs.world_to_pixel(ra, dec)
            assert np.hypot(x - corners[:, 0], y - corners[:, 1]).max() < 0.3     # pixels
            assert abs(row["orientation_deg"] - M.TRUE_ORIENTATION_DEG) < 0.2 and row["parity"] == 1
            assert row["segment"] == 1 and row["header"]["CTYPE1"] == "RA---TAN"
    # The calibrated images are not modified: no WCS keyword added.
    out = calibrated(wd, next(iter(groups.values())))[0]
    assert "CRVAL1" not in fits.getheader(out)
    assert (wd / "astrometry" / "astrometry.md").exists()
    qa = load(wd / "qa" / "qa.json")
    assert qa["astrometry"] == [] and len(qa["tables"]["astrometry"]) == 4


@pytest.mark.siril
def test_golden_run(siril_run, minisky, golden):
    res, wd, plan, groups = siril_run
    repl = minisky.replacements()
    aliases = master_aliases(plan)
    images = {}
    for (label, filt), g in sorted(groups.items()):
        for out in calibrated(wd, g):
            sig = image_signature(out)
            images[f"{label}/{filt}/{out.name}"] = {k: v for k, v in sig.items() if k != "sha256"}
    golden.check("run", {"exit_code": res.exit_code, "stdout": golden.text(res.output, repl),
                         "plan.json": plan, "qa.json": load(wd / "qa" / "qa.json"),
                         "masters.md": golden.text((wd / "masters" / "masters.md").read_text(encoding="utf-8"),
                                                   repl),
                         "calibrated": images}, repl, aliases)


@pytest.mark.siril
def test_run_raw_untouched_and_idempotent(siril_run, minisky):
    # Last test of the module: it rewrites the outputs of the first run.
    _, wd, _, _ = siril_run
    assert M.file_state(minisky.root) == minisky.raw_state
    again = invoke("run", minisky.root, "-c", minisky.config, "-w", wd)
    assert again.exit_code == 0, again.output
    assert "4 groupe(s) déjà à jour" in again.output
    assert M.file_state(minisky.root) == minisky.raw_state
