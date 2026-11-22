"""Raw pixel checks (P*) from per-frame statistics in memory (no FITS)."""

import math

import pytest

from builders import inventory_of, make_frame_stats, sept06_frames, series_frames, two_nights_frames
from varstar.domain.model import DARK, FLAT, LIGHT
from varstar.domain.qa.raw import (flat_comparison_series, flat_dark_level, flat_pair_check, frames_to_measure,
                                   raw_pixel_checks, sample_lights, saturation_adu)


def stats(frames, rule):
    """``rule(frame)`` → statistics dict, keyed like the PixelStats port."""
    return {str(f.path): rule(f) for f in frames}


def by_folder(levels, default=600.0, sigma=12.0):
    def rule(f):
        for prefix, v in levels.items():
            if f.rel_path.startswith(prefix):
                return make_frame_stats(**v) if isinstance(v, dict) else make_frame_stats(v, sigma)
        return make_frame_stats(default, sigma)
    return rule


def run(frames, rule):
    inv = inventory_of(frames)
    return inv, raw_pixel_checks(inv, inv.config, stats(inv.frames, rule))


def checks(res, check):
    return [i for i in res.items if i.check == check]


def test_frames_to_measure_and_sample_lights():
    inv = inventory_of(sept06_frames() + series_frames("light/H/SESSION_sept06", LIGHT, 300,
                                                       "2021-09-07T01:00:00", 4, step_s=600, filter="H"))
    lights = sample_lights(inv)
    assert len(lights) == 6 and len({f.rel_path for f in lights}) == 6         # first, middle, last per group
    measured = frames_to_measure(inv)
    assert sum(f.imagetyp == LIGHT for f in measured) == 6
    assert sum(f.imagetyp != LIGHT for f in measured) == 6 + 4 + 6
    assert saturation_adu(inv) == 65532


def test_offset_jump_is_p1():
    inv, res = run(two_nights_frames(), by_folder({"dark/a": 516.0, "dark/b": 541.0}))
    (p1,) = checks(res, "P1")
    assert p1.data["classification"] == "offset" and p1.data["delta_adu"] == 25
    assert p1.data["sets"] == ["DARK-300s-20210907"] and len(p1.files) == 5
    assert not checks(res, "P2")
    levels = {r["set"]: r["level_adu"] for r in res.tables["dark_levels"]}
    assert levels == {"DARK-300s-20210902": 516.0, "DARK-300s-20210907": 541.0}
    # P8 table: sept06 lights sit 59 ADU above their own darks, sept02 84 ADU.
    sky = {r["group"]: r["raw_sky_adu"] for r in res.tables["raw_sky"]}
    assert sky == {"sept02 / H": 84.0, "sept06 / H": 59.0} and not checks(res, "P8")


def test_extra_shot_noise_is_a_light_leak_p2():
    leak = {"level": 541.0, "sigma": math.sqrt(144 + 100)}
    _, res = run(two_nights_frames(), by_folder({"dark/a": 516.0, "dark/b": leak}))
    assert [i.check for i in res.items if i.check in ("P1", "P2")] == ["P2"]
    assert res.items[0].data["classification"] == "lumière"


def test_non_uniform_excess_is_a_light_leak_p2():
    corner = {"level": 541.0, "sigma": 12.0, "corners": {"haut_gauche": 30.0, "haut_droit": 0.0,
                                                         "bas_gauche": 0.0, "bas_droit": 0.0}}
    _, res = run(two_nights_frames(), by_folder({"dark/a": 516.0, "dark/b": corner}))
    (p2,) = checks(res, "P2")
    assert p2.data["classification"] == "offset" and p2.data["corner_diff_adu"] == 30


def test_single_odd_dark_in_a_series_is_p2():
    frames = two_nights_frames()
    odd = next(f for f in frames if f.rel_path.startswith("dark/b"))
    _, res = run(frames, lambda f: make_frame_stats(526.0 if f.rel_path == odd.rel_path else 516.0))
    (p2,) = checks(res, "P2")
    assert p2.files == [odd.rel_path] and p2.data == {"max_delta_adu": 10.0}


def test_flat_exposure_p3():
    def rule(f):
        if f.imagetyp == FLAT:
            return make_frame_stats(60000.0 if f.filter == "S" else 30000.0, 100.0)
        return make_frame_stats(516.0)
    _, res = run(sept06_frames(), rule)
    (p3,) = checks(res, "P3")
    assert "FLAT" in p3.message and len(p3.files) == 3
    assert {r["filter"]: r["level_max_frac"] for r in res.tables["flat_levels"]} == {"H": 0.458, "S": 0.916}


def test_adu_scale_change_p7():
    frames = sept06_frames()
    odd = next(f for f in frames if f.imagetyp == DARK)
    _, res = run(frames, lambda f: make_frame_stats(516.0, adu_scale=2 if f.rel_path == odd.rel_path else 4))
    (p7,) = checks(res, "P7")
    assert p7.files == [odd.rel_path] and "×4" in p7.message


@pytest.mark.parametrize("light, severity", [(520.0, "error"), (546.0, "warning"), (600.0, None)])
def test_lights_below_their_darks_p8(light, severity):
    frames = [f for f in two_nights_frames() if not f.rel_path.startswith(("dark/a", "light/H/SESSION_sept02"))]
    _, res = run(frames, by_folder({"dark/": 545.0}, default=light))
    assert [i.severity for i in checks(res, "P8")] == ([severity] if severity else [])


def test_flat_comparison_series_and_pair_check():
    frames = sept06_frames() + series_frames("flat/SESSION_sept10", FLAT, 1.13, "2021-09-10T13:00:00", 3,
                                             step_s=2, filter="H")
    inv = inventory_of(frames)
    series_ = flat_comparison_series(inv)
    assert [(filt, len(sets)) for filt, sets in series_] == [("H", 2)]
    a, b = series_[0][1]
    assert a.start < b.start
    summaries = {s.id: {"level": 100.0} for s in inv.calibration_sets["DARKFLAT"]}
    assert flat_dark_level(a, inv, inv.config, summaries) == 100.0

    item, row = flat_pair_check(inv, inv.config, "H", a, b, {"dust_pct": 0.2, "shape_pct": 0.3, "noise_pct": 0.1},
                                "/w/qa/flats/r.fits")
    assert item is None and row["verdict"] == "identiques" and row["a"] == a.id
    item, row = flat_pair_check(inv, inv.config, "H", a, b, {"dust_pct": 2.0, "shape_pct": 3.0, "noise_pct": 0.1},
                                "/w/qa/flats/r.fits")
    assert item.severity == "warning" and item.files == ["/w/qa/flats/r.fits"] and item.data is row
    assert row["verdict"] == "poussières déplacées et forme différente (vignetage)"
    item, _ = flat_pair_check(inv, inv.config, "H", a, b, {"dust_pct": 0.2, "shape_pct": 3.0, "noise_pct": 0.1}, "r")
    assert item.severity == "info"


def test_flat_dark_level_without_dark_of_flat_is_zero():
    inv = inventory_of([f for f in sept06_frames() if not f.rel_path.startswith("dark/")])
    (s, *_) = inv.calibration_sets[FLAT]
    assert flat_dark_level(s, inv, inv.config, {}) == 0.0
