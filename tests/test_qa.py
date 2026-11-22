"""Early quality checks (docs/QA_CHECKS.md) and the same-night dark policy (D16)."""

import numpy as np
from typer.testing import CliRunner

from varstar import qa
from varstar.cli import app
from varstar.inventory import build_inventory

from fitsgen import series

SHAPE = (128, 160)


def noise(level, sigma, seed, scale=4, extra=None):
    rng = np.random.default_rng(seed)
    adu14 = rng.normal(level / scale, sigma / scale, SHAPE)
    if extra is not None:
        adu14 = adu14 + extra / scale
    return (np.clip(np.round(adu14), 0, 16383) * scale).astype(np.uint16)


def two_nights(root, level2=516.0, leak=False, temp=-12.0):
    """sept02-like night with darks at 516 ADU and a sept06-like night with darks at `level2`."""
    series(root, "light/H/SESSION_sept02", "LIGHT", 300, "2021-09-02T23:30:00", 3, step_s=600, filt="H",
           shape=SHAPE, make=lambda i: noise(600, 12, 100 + i))
    series(root, "light/H/SESSION_sept06", "LIGHT", 300, "2021-09-06T23:04:00", 3, step_s=600, filt="H",
           shape=SHAPE, make=lambda i: noise(600, 12, 200 + i), temp=temp)
    series(root, "dark/a", "DARK", 300, "2021-09-02T20:00:00", 5, step_s=310, prefix="", shape=SHAPE,
           make=lambda i: noise(516, 12, 300 + i))

    def night2(i):
        rng = np.random.default_rng(400 + i)
        if leak:  # extra photons: Poisson noise at gain 0.25 e-/ADU
            light = rng.poisson(25 * 0.25, SHAPE) / 0.25
            return noise(516, 12, 400 + i, extra=light)
        return noise(level2, 12, 400 + i)

    series(root, "dark/b", "DARK", 300, "2021-09-07T02:00:00", 5, step_s=310, prefix="", shape=SHAPE, make=night2)
    return root


import pytest


@pytest.fixture(autouse=True)
def full_pixels(cfg):
    cfg["qa"]["pixel_step"] = 1  # small synthetic frames: use every pixel
    cfg["runtime"]["processes"] = 1


def test_same_night_darks_only(tmp_path, cfg):
    root = two_nights(tmp_path / "Y")
    series(root, "light/H/SESSION_sept12", "LIGHT", 300, "2021-09-12T23:00:00", 2, step_s=600, filt="H",
           shape=SHAPE)
    inv = build_inventory(root, cfg)
    by = {g.label: g for g in inv.light_groups}
    assert by["sept02"].dark.level == "A" and [s.id for s in by["sept02"].dark.sets] == ["DARK-300s-20210902"]
    assert by["sept06"].dark.level == "A" and [s.id for s in by["sept06"].dark.sets] == ["DARK-300s-20210907"]
    # No dark that night: the nearest single night, level B, never a mix.
    g12 = by["sept12"]
    assert g12.dark.level == "B" and [s.id for s in g12.dark.sets] == ["DARK-300s-20210907"]
    assert "même nuit" in g12.dark.reasons[-1]
    cfg["matching"]["dark_same_night"] = False
    inv = build_inventory(root, cfg)
    assert len(next(g for g in inv.light_groups if g.label == "sept12").dark.sets) == 2


def test_offset_jump_detected(tmp_path, cfg):
    root = two_nights(tmp_path / "Y", level2=541.0)
    inv = build_inventory(root, cfg)
    items, tables = qa.raw_checks(inv, cfg, tmp_path / "work")
    p1 = [i for i in items if i.check == "P1"]
    assert len(p1) == 1 and p1[0].data["classification"] == "offset"
    assert abs(p1[0].data["delta_adu"] - 25) < 2
    assert not [i for i in items if i.check == "P2"]
    levels = {r["set"]: r["level_adu"] for r in tables["dark_levels"]}
    assert levels["DARK-300s-20210907"] - levels["DARK-300s-20210902"] > 20
    # P8: the sept06 lights (600 ADU) sit only ~59 ADU above their dark; sept02 ~84 ADU.
    sky = {r["group"]: r["raw_sky_adu"] for r in tables["raw_sky"]}
    assert sky["sept06 / H"] < sky["sept02 / H"]


def test_light_leak_detected(tmp_path, cfg):
    root = two_nights(tmp_path / "Y", leak=True)
    inv = build_inventory(root, cfg)
    items, _ = qa.raw_checks(inv, cfg, tmp_path / "work")
    assert [i.check for i in items if i.check in ("P1", "P2")] == ["P2"]


def test_classify_excess():
    assert qa.classify_excess(25, 12, 12, 0.25) == "offset"
    assert qa.classify_excess(25, 12, float(np.sqrt(144 + 100)), 0.25) == "lumière"
    assert qa.classify_excess(25, 12, 12, None) == "indéterminé"


def test_consistent_dark_sets_keeps_nearest(tmp_path, cfg):
    root = two_nights(tmp_path / "Y", level2=541.0)
    cfg["matching"]["dark_same_night"] = False
    inv = build_inventory(root, cfg)
    g = next(g for g in inv.light_groups if g.label == "sept06")
    cache = qa.StatsCache(tmp_path / "work", cfg, 65532)
    stats = cache.get([f for s in g.dark.sets for f in s.frames])
    kept, note = qa.consistent_dark_sets(g.dark.sets, stats, 5.0, g.start)
    assert [s.id for s in kept] == ["DARK-300s-20210907"] and "niveaux différents" in note
    # The cache is reused (same ids) on a second call.
    assert qa.StatsCache(tmp_path / "work", cfg, 65532).get(g.dark.sets[0].frames)


def test_dark_above_lights_is_error(tmp_path, cfg):
    root = tmp_path / "Y"
    series(root, "light/H/SESSION_s", "LIGHT", 300, "2021-09-06T23:00:00", 3, step_s=600, filt="H", shape=SHAPE,
           make=lambda i: noise(520, 12, i))
    series(root, "dark/x", "DARK", 300, "2021-09-07T02:00:00", 4, step_s=310, prefix="", shape=SHAPE,
           make=lambda i: noise(545, 12, 50 + i))
    items, _ = qa.raw_checks(build_inventory(root, cfg), cfg, tmp_path / "work")
    p8 = [i for i in items if i.check == "P8"]
    assert p8 and p8[0].severity == "error"


def test_header_checks_temperature_and_dew(tmp_path, cfg):
    root = two_nights(tmp_path / "Y", temp=-9.5)
    series(root, "light/H/SESSION_sept12", "LIGHT", 300, "2021-09-12T23:00:00", 2, step_s=600, filt="H",
           shape=SHAPE, extra={"AMBTEMP": 10.0, "DEWPOINT": 9.0, "CLOUDCVR": 80.0})
    items = qa.header_checks(build_inventory(root, cfg), cfg)
    h3 = [i for i in items if i.check == "H3"]
    assert h3 and h3[0].data["n_bad"] == 3 and "toute la nuit" in h3[0].message
    checks = {(i.check, i.severity) for i in items}
    assert ("H7", "warning") in checks and ("H7", "info") in checks
    assert ("H9", "info") in checks  # 5 darks < 15


def test_flat_ratio_metrics():
    yy, xx = np.mgrid[0:200, 0:300]
    vign = 1 - 0.3 * (((yy - 100) / 150) ** 2 + ((xx - 150) / 200) ** 2)

    def dust(cy, cx):
        r = np.hypot(yy - cy, xx - cx)
        return 1 - 0.04 * ((r > 4) & (r < 8))

    rng = np.random.default_rng(1)
    a = (vign * dust(60, 90) * (1 + 0.0005 * rng.standard_normal(vign.shape))).astype(np.float32)
    b = (vign * dust(60, 90) * (1 + 0.0005 * rng.standard_normal(vign.shape))).astype(np.float32)
    c = (vign * dust(130, 200) * (1 + 0.0005 * rng.standard_normal(vign.shape))).astype(np.float32)
    same = qa.flat_ratio_metrics(a, b)
    moved = qa.flat_ratio_metrics(a, c)
    assert same["dust_pct"] < 0.5 and same["shape_pct"] < 0.5
    assert moved["dust_pct"] > 2.0


def test_check_command(tmp_path):
    root = two_nights(tmp_path / "Y", level2=541.0)
    res = CliRunner().invoke(app, ["check", str(root), "-w", str(tmp_path / "w")])
    assert res.exit_code == 0, res.output
    assert "P1" in res.output and "offset" in res.output
    md = (tmp_path / "w" / "qa" / "qa.md").read_text()
    assert "Niveau des darks par nuit" in md and "Conseil pour la prochaine séance" in md
    res = CliRunner().invoke(app, ["check", str(root), "-w", str(tmp_path / "w"), "--headers-only"])
    assert res.exit_code == 0 and "[P1" not in res.output


def test_borrowed_darks_avoid_anomalous_night(tmp_path, cfg):
    root = two_nights(tmp_path / "Y", level2=541.0)
    series(root, "dark/c", "DARK", 300, "2021-09-01T01:00:00", 5, step_s=310, prefix="", shape=SHAPE,
           make=lambda i: noise(516, 12, 500 + i))
    series(root, "light/H/SESSION_sept12", "LIGHT", 300, "2021-09-12T23:00:00", 2, step_s=600, filt="H",
           shape=SHAPE, make=lambda i: noise(700, 12, 600 + i))
    inv = build_inventory(root, cfg)
    g = next(g for g in inv.light_groups if g.label == "sept12")
    assert [s.id for s in g.dark.sets] == ["DARK-300s-20210907"]  # nearest night (header-only matching)
    stats = qa.StatsCache(tmp_path / "work", cfg, 65532).get([f for s in qa.dark_candidates(inv, g, cfg)
                                                             for f in s.frames])
    sets, notes = qa.effective_dark_sets(inv, g, stats, cfg)
    assert [s.id for s in sets] == ["DARK-300s-20210902"] and "offset anormal" in notes[0]
    # The anomalous night keeps its own darks.
    g6 = next(g for g in inv.light_groups if g.label == "sept06")
    assert [s.id for s in qa.effective_dark_sets(inv, g6, stats, cfg)[0]] == ["DARK-300s-20210907"]


def test_ampglow_residual_needs_all_filters():
    rows = [{"group": "g1", "label": "s / H", "session": "s", "dark_id": "d", "ampglow_zone_resid_adu": [-24.5, 1.0]},
            {"group": "g2", "label": "s / S", "session": "s", "dark_id": "d", "ampglow_zone_resid_adu": [-5.3, 0.5]}]
    items = qa._ampglow_residual_items(rows)
    assert [(i.check, i.severity) for i in items] == [("C1", "info")]
    rows[1]["ampglow_zone_resid_adu"] = [-20.0, 0.5]
    assert qa._ampglow_residual_items(rows)[0].severity == "warning"
