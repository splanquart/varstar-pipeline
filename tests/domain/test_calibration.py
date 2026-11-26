"""Calibration decisions (strategy B): rejection, master identity, dark mode, masks, groups, statistics."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from builders import inventory_of, make_frame, make_frames, make_settings, sept06_frames, series_frames
from varstar.domain.calibration import (FileStat, MasterRecord, PipelineError, TemporalNoise, calibrated_name,
                                        calibrated_subdir, calibration_fingerprint, choose_rejection,
                                        compute_master_id, corner_boxes, crosscheck_stats, dark_consistency,
                                        dark_label, dark_mode, dark_params, flat_label, flat_level_warnings,
                                        flat_params, group_key, mask_ids, master_frame_stats, master_summary,
                                        offset_dark_sets, safe_stem, select_groups, unstable_mask)
from varstar.domain.model import DARK, DARKFLAT, FLAT, LIGHT

STATS = [FileStat(f"/raw/d{i}.fits", 1000 + i, 10 ** 9 * i) for i in range(5)]


@pytest.mark.parametrize("n,code", [(3, "p"), (7, "p"), (8, "s"), (14, "s"), (15, "w"), (25, "w"), (26, "l"),
                                    (65, "l")])
def test_choose_rejection(n, code):
    rej = choose_rejection(n, make_settings().masters)
    assert rej.code == code
    assert rej.siril_args().startswith(f"rej {code} ")
    assert rej.to_dict()["name"] == rej.name


@given(st.permutations(STATS))
def test_master_id_ignores_the_order_of_the_frames(stats):
    params = dark_params(choose_rejection(5, make_settings().masters))
    assert compute_master_id("dark", stats, params) == compute_master_id("dark", STATS, params)


def test_master_id_changes_with_any_input():
    rej = choose_rejection(5, make_settings().masters)
    ref = compute_master_id("dark", STATS, dark_params(rej))
    assert len(ref) == 16
    touched = STATS[:4] + [FileStat(STATS[4].resolved, STATS[4].size, STATS[4].mtime_ns + 1)]
    assert compute_master_id("dark", touched, dark_params(rej)) != ref
    assert compute_master_id("darkflat", STATS, dark_params(rej)) != ref
    assert compute_master_id("dark", STATS[:4], dark_params(rej)) != ref
    assert compute_master_id("dark", STATS, dark_params(choose_rejection(10, make_settings().masters))) != ref
    assert compute_master_id("flat", STATS, flat_params(rej, "a")) != compute_master_id("flat", STATS,
                                                                                         flat_params(rej, "b"))


def test_labels():
    darks = make_frames(3, DARK, exptime=300, filter=None, ccd_temp=-12.2)
    assert dark_label("dark", darks) == "DARK_g120_ox_b2_300s_T-12"
    assert dark_label("darkflat", make_frames(2, DARKFLAT, exptime=1.13, filter=None, ccd_temp=None)) == \
        "DARKFLAT_g120_ox_b2_1.13s"
    flats = make_frames(3, FLAT, dt.datetime(2021, 9, 7, 13, 0), exptime=1.13)
    assert flat_label(flats) == "FLAT_g120_ox_b2_1.13s_H_20210907"


def test_dark_mode():
    inv = inventory_of(sept06_frames())
    darks = inv.calibration_sets[DARK]
    assert dark_mode(darks, 300.0, 0.01) == "direct"
    assert dark_mode(darks, 600.0, 0.01) == "scaled"


def test_offset_dark_sets_prefers_the_shortest():
    inv = inventory_of(sept06_frames())
    g = inv.light_groups[0]
    kind, sets = offset_dark_sets(inv, g)
    assert kind == "darkflat" and {s.exptime for s in sets} == {1.13}
    no_darks = inventory_of([f for f in sept06_frames() if not f.rel_path.startswith("dark/")])
    with pytest.raises(PipelineError, match="aucun dark court"):
        offset_dark_sets(no_darks, no_darks.light_groups[0])


def test_mask_ids():
    assert mask_ids("d1", "f1", "abcd") == ("bpm_d1_f1_abcd.fits", "ampglow_d1_abcd.fits")
    assert mask_ids(None, "f1", "abcd") == ("bpm_nodark_f1_abcd.fits", None)
    assert mask_ids("d1", None, "abcd")[0] == "bpm_d1_noflat_abcd.fits"


def test_group_selection():
    frames = sept06_frames() + series_frames("light/H/SESSION_sept12", LIGHT, 300, "2021-09-12T23:00:00", 2,
                                             step_s=600, filter="H")
    inv = inventory_of(frames)
    assert [group_key(g) for g in inv.light_groups][:2] == ["2021-09-06_H_300s", "2021-09-06_S_300s"]
    assert len(select_groups(inv, None, None)) == 3
    assert [g.label for g in select_groups(inv, ["sept12"], None)] == ["sept12"]
    assert [g.filter for g in select_groups(inv, ["2021-09-06"], ["S"])] == ["S"]
    with pytest.raises(PipelineError, match="inconnue"):
        select_groups(inv, ["sept99"], None)


def test_calibrated_names():
    assert safe_stem("2021-09-06_23-04-00_NGC 1499_H_300.00s.fits") == "2021-09-06_23-04-00_NGC_1499_H_300.00s"
    assert calibrated_name("/raw/light/a b (1).fits") == "a_b_1.fits"
    assert calibrated_subdir("2021-09-06", "sept06", "H") == "calibrated/2021-09-06_sept06/H"
    assert calibrated_subdir("2021-09-06", None, "S") == "calibrated/2021-09-06/S"


def test_calibration_fingerprint():
    group = {"level": "A", "dark": {"id": "d"}, "flat": None, "darkflat": None, "bpm": "b", "ampglow": None}
    raws = [("/raw/a.fits", FileStat("/raw/a.fits", 10, 1))]
    ref = calibration_fingerprint(group, raws)
    assert calibration_fingerprint({**group, "files": ["ignored"]}, raws) == ref
    assert calibration_fingerprint({**group, "level": "B"}, raws) != ref
    assert calibration_fingerprint({**group, "bpm": "c"}, raws) != ref
    assert calibration_fingerprint(group, [("/raw/a.fits", FileStat("/raw/a.fits", 10, 2))]) != ref


# --------------------------------------------------------------------------- statistics of the source frames


def test_master_frame_stats_and_corners():
    data = np.full((40, 60), 500, np.float32)
    data[:10, :10] += 30
    s = master_frame_stats(make_frame(DARK, rel_path="dark/a.fits"), data)
    assert s["median"] == 500 and s["corners"]["haut_gauche"] == 30 and s["corners"]["bas_droit"] == 0
    assert s["file"] == "dark/a.fits" and s["frac_saturation"] == pytest.approx(500 / 65532)
    assert corner_boxes((40, 60))["bas_droit"] == (slice(30, 40), slice(50, 60))


def test_temporal_noise_unstable_ignores_single_cosmic():
    rng = np.random.default_rng(2)
    noise = TemporalNoise()
    for i in range(10):
        d = 500 + rng.normal(0, 8, (40, 40))
        d[10, 10] = 500 + (300 if i % 2 else -300)    # RTS-like pixel, unstable
        if i == 3:
            d[20, 20] = 5000                            # single cosmic ray
        noise.add((np.round(d / 4) * 4).astype(np.float32))
    extra = noise.result()
    assert 6 < extra["sigma_single"] < 10
    pstd = extra["pixel_std"]
    med = np.median(pstd)
    assert pstd[10, 10] > 5 * med and pstd[20, 20] < 3 * med
    mask, med_std = unstable_mask(pstd, 5.0)
    assert mask.dtype == np.uint8 and mask[10, 10] == 1 and mask[20, 20] == 0 and med_std > 0


def test_temporal_noise_with_few_frames():
    two = TemporalNoise()
    two.add(np.zeros((8, 8), np.float32))
    two.add(np.ones((8, 8), np.float32))
    assert set(two.result()) == {"sigma_single"}
    assert TemporalNoise().result() == {}


def test_flat_level_warnings():
    stats = [{"file": "a", "frac_saturation": 0.5}, {"file": "b", "frac_saturation": 0.9},
             {"file": "c", "frac_saturation": 0.1}]
    assert flat_level_warnings(stats, make_settings().masters) == ["b", "c"]


def test_dark_consistency_by_folder():
    corners = {"haut_gauche": 1.0, "haut_droit": 0.0, "bas_gauche": 0.0, "bas_droit": 0.0}
    stats = [{"file": f"dark/{d}/x{i}.fits", "median": m, "robust_sigma": 12.0, "corners": corners,
              "ccd_temp": -12.0} for d, m in (("a", 516.0), ("b", 541.0)) for i in range(2)]
    rows = dark_consistency(stats)
    assert [(r["folder"], r["n"], r["median_adu"]) for r in rows] == [("dark/a", 2, 516.0), ("dark/b", 2, 541.0)]
    assert rows[0]["corner_excess_adu"]["haut_gauche"] == 1.0


def test_crosscheck_detects_a_flipped_reference():
    rng = np.random.default_rng(0)
    ours = (500 + rng.normal(0, 5, (40, 60))).astype(np.float32)
    same = crosscheck_stats("dark", ours, ours + 2)
    assert same["flipped"] is False and same["median_diff_adu"] == pytest.approx(-2)
    assert crosscheck_stats("dark", ours, (ours + 2)[::-1])["flipped"] is True
    flat = crosscheck_stats("flat", ours, ours * 3)
    assert abs(flat["median_ratio_pct"]) < 1e-3 and flat["robust_sigma_ratio_pct"] < 1e-3
    assert "dimensions différentes" in crosscheck_stats("dark", ours, ours[:10])["error"]


def test_master_summary():
    rec = MasterRecord("abc", "flat", Path("/lib/flats/F_abc"), {
        "label": "F", "n_frames": 5, "rejection": {"name": "rejet par percentile"}, "median_adu": 30000.0,
        "exptime": 1.13, "filter": "H", "level_min_frac": 0.4, "level_max_frac": 0.5})
    s = master_summary(rec)
    assert s["dir"] == "/lib/flats/F_abc" and s["rejection"] == "rejet par percentile"
    assert s["sigma_single_frame_adu"] is None and s["level_max_frac"] == 0.5
    assert rec.path.name == "master.fits" and rec.unstable_path.name == "unstable.fits"
