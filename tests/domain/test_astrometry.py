"""Astrometry, pure part: projection, WCS solutions, plate solving, segments."""

import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from builders import make_solution, synthetic_sky
from varstar.domain.astrometry import (CatalogStars, SolveHint, SolveJob, SolveResult, SolveSettings, WcsSolution,
                                       angle_diff_deg, angular_sep_arcsec, assign_segments, camera_angle_deg,
                                       catalog_cone, circular_median_deg, expected_scale_arcsec, radec_to_std,
                                       solve_field, solve_image, solve_series, std_to_radec)
from varstar.domain.stars import Stars, detect_stars

W, H = 600, 400
SCALE = 3.4877


# =========================================================================== projection


@settings(max_examples=200, deadline=None)
@given(ra0=st.floats(0, 360, exclude_max=True), dec0=st.floats(-89, 89),
       dra=st.floats(-3, 3), ddec=st.floats(-3, 3))
@example(ra0=359.9, dec0=0.0, dra=0.5, ddec=0.0)        # RA wrap-around
@example(ra0=10.0, dec0=88.0, dra=3.0, ddec=0.9)        # near the pole
def test_projection_round_trip(ra0, dec0, dra, ddec):
    dec = max(-89.9, min(89.9, dec0 + ddec))
    ra = (ra0 + dra) % 360
    xi, eta = radec_to_std(ra, dec, ra0, dec0)
    ra2, dec2 = std_to_radec(xi, eta, ra0, dec0)
    assert float(angular_sep_arcsec(ra, dec, ra2, dec2)) < 1e-6


def test_projection_basics():
    xi, eta = radec_to_std(10.0, 20.0, 10.0, 20.0)
    assert abs(xi) < 1e-12 and abs(eta) < 1e-12
    xi, eta = radec_to_std(10.0, 21.0, 10.0, 20.0)       # north → +eta
    assert abs(xi) < 1e-9 and 0.99 < eta < 1.01
    xi, _ = radec_to_std(11.0, 20.0, 10.0, 20.0)         # east → +xi
    assert xi > 0
    xi, eta = radec_to_std(190.0, 0.0, 10.0, 0.0)        # opposite hemisphere: not projectable
    assert np.isnan(xi) and np.isnan(eta)


def test_expected_scale_uses_binned_xpixsz():
    # XPIXSZ of N.I.N.A. already includes the binning: 4.63 µm at 274 mm → 3.49″/px.
    assert abs(expected_scale_arcsec(274.0, 4.63) - 3.4855) < 1e-3
    assert expected_scale_arcsec(None, 4.63) is None and expected_scale_arcsec(274.0, 0) is None


# =========================================================================== WCS solutions


@pytest.mark.parametrize("orientation,parity", [(0.0, -1), (37.0, -1), (180.6, 1), (271.0, 1)])
def test_solution_geometry(orientation, parity):
    sol = make_solution(scale_arcsec=SCALE, width=W, height=H, orientation_deg=orientation, parity=parity)
    assert abs(sol.scale_arcsec - SCALE) < 1e-9
    assert abs(angle_diff_deg(sol.orientation_deg, orientation)) < 1e-6
    assert sol.parity == parity and sol.skew < 1e-12
    # One pixel towards north on the grid moves north on the sky.
    t = math.radians(orientation)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    _, d0 = sol.pixel_to_world(cx, cy)
    _, d1 = sol.pixel_to_world(cx + 10 * math.sin(t), cy + 10 * math.cos(t))
    assert abs((d1 - d0) * 3600 - 10 * SCALE) < 1e-3


def test_solution_round_trip_with_sip_and_serialisation():
    base = make_solution(scale_arcsec=SCALE, width=W, height=H, orientation_deg=12.0)
    sol = WcsSolution(base.crval, base.crpix, base.cd, 3, ((2, 0, 1e-6), (1, 2, -2e-9)), ((0, 2, 3e-7), (3, 0, 1e-9)))
    x = np.array([0.0, 100.0, W - 1.0, 250.5])
    y = np.array([0.0, 300.0, H - 1.0, 10.25])
    ra, dec = sol.pixel_to_world(x, y)
    x2, y2 = sol.world_to_pixel(ra, dec)
    assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)
    again = WcsSolution.from_dict(sol.to_dict())
    assert again == sol
    h = sol.header()
    assert h["CTYPE1"] == "RA---TAN-SIP" and h["A_ORDER"] == 3 and h["A_2_0"] == 1e-6 and h["B_3_0"] == 1e-9
    assert make_solution().header()["CTYPE2"] == "DEC--TAN" and "A_ORDER" not in make_solution().header()


def test_solve_result_round_trip():
    res = SolveResult(True, make_solution(), None, 300, 900, 280, 0.3, {"vote": 40})
    d = res.to_dict()
    assert d["parity"] == -1 and abs(d["scale_arcsec"] - 3.4877) < 1e-9 and d["header"]["CRVAL1"] == 60.44
    assert SolveResult.from_dict(d) == res
    bad = SolveResult(False, reason="nuages")
    assert SolveResult.from_dict(bad.to_dict()) == bad


# =========================================================================== synthetic sky


def sky(truth: WcsSolution, n: int = 900, seed: int = 1, noise: float = 8.0):
    return synthetic_sky(truth, (H, W), n=n, seed=seed, noise=noise)


@pytest.mark.parametrize("orientation,parity", [(0.4, -1), (180.6, 1), (63.0, 1)])
def test_solve_field_recovers_the_wcs(orientation, parity):
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=orientation,
                          parity=parity)
    img, cat = sky(truth)
    hint = SolveHint(60.44 + 0.05, 36.11 - 0.04, SCALE * 1.015, W, H)    # header 4′ off, scale 1.5 % off
    res = solve_image(img, cat, hint, SolveSettings(sip_order=0, min_matches=15))
    assert res.ok, res.reason
    sol = res.solution
    assert res.n_matched >= 30 and res.rms_arcsec < 1.0
    assert sol.parity == parity and angle_diff_deg(sol.orientation_deg, orientation) < 0.05
    assert abs(sol.scale_arcsec / SCALE - 1) < 1e-3
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 5), np.linspace(0, H - 1, 5))
    ra, dec = truth.pixel_to_world(gx.ravel(), gy.ravel())
    x, y = sol.world_to_pixel(ra, dec)
    assert np.hypot(x - gx.ravel(), y - gy.ravel()).max() < 0.3
    assert res.stats["background_adu"] > 900


def test_solve_field_fits_the_distortion():
    base = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=20.0)
    truth = WcsSolution(base.crval, base.crpix, base.cd, 3, ((3, 0, 2e-7), (1, 2, 2e-7)), ((0, 3, 2e-7), (2, 1, 2e-7)))
    img, cat = sky(truth, n=1500, seed=3)
    hint = SolveHint(60.44, 36.11, SCALE, W, H)
    linear = solve_image(img, cat, hint, SolveSettings(sip_order=0, max_rms_arcsec=10, min_matches=15))
    cubic = solve_image(img, cat, hint, SolveSettings(sip_order=3, min_matches=15))
    assert cubic.ok and cubic.solution.sip_order == 3
    assert cubic.rms_arcsec < 0.5 * linear.rms_arcsec


def test_prior_narrows_the_search_and_falls_back():
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=180.6, parity=1)
    img, cat = sky(truth)
    hint = SolveHint(60.44, 36.11, SCALE, W, H)
    s = SolveSettings(sip_order=0, min_matches=15)
    first = solve_image(img, cat, hint, s)
    narrowed = hint.with_prior(first)
    assert narrowed.parity == 1 and narrowed.angle_deg is not None
    assert solve_image(img, cat, narrowed, s).ok
    # Meridian flip in the series: the prior is wrong by 180°, the full search still finds it.
    flipped = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=0.6, parity=1)
    img2, _ = sky(flipped)
    res = solve_image(img2, cat, narrowed, s)
    assert res.ok and angle_diff_deg(res.solution.orientation_deg, 0.6) < 0.1
    assert hint.with_prior(SolveResult(False, reason="x")) is hint


def test_solve_field_failures():
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H)
    img, cat = sky(truth)
    hint = SolveHint(60.44, 36.11, SCALE, W, H)
    s = SolveSettings(sip_order=0, min_matches=15)
    few = solve_field(Stars(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([5.0, 4.0])), cat, hint, s)
    assert not few.ok and "étoile(s) détectée(s)" in few.reason
    stars = detect_stars(img)
    empty = CatalogStars(np.array([200.0]), np.array([-30.0]), np.array([8.0]))
    assert "catalogue" in solve_field(stars, empty, hint, s).reason
    elsewhere = solve_field(stars, cat, SolveHint(61.6, 37.0, SCALE, W, H), s)      # pointing 1° off
    assert not elsewhere.ok
    wrong_scale = solve_field(stars, cat, SolveHint(60.44, 36.11, SCALE * 1.5, W, H), s)
    assert not wrong_scale.ok
    noisy = SolveSettings(sip_order=0, min_matches=15, max_rms_arcsec=0.01)
    res = solve_field(stars, cat, hint, noisy)
    assert not res.ok and "résidus" in res.reason and res.solution is not None


def test_solve_series_orders_and_isolates_failures():
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=5.0)
    img, cat = sky(truth)
    hint = SolveHint(60.44, 36.11, SCALE, W, H)
    seen = []

    def load(path):
        seen.append(path)
        if path == "broken":
            raise OSError("disque")
        return img

    jobs = [SolveJob("b", hint, "2021-09-07T01:00:00"), SolveJob("a", hint, "2021-09-07T00:00:00"),
            SolveJob("broken", hint, "2021-09-07T02:00:00"), SolveJob("nohint", None, None, "RA/DEC absents")]
    out = solve_series(jobs, load, cat, SolveSettings(sip_order=0, min_matches=15))
    assert seen == ["a", "b", "broken"]
    assert out["a"].ok and out["b"].ok
    assert "illisible" in out["broken"].reason and out["nohint"].reason == "RA/DEC absents"
    assert "vote_angle_deg" in out["b"].stats


def test_catalog_cone_covers_every_image():
    hints = [SolveHint(60.0, 36.0, SCALE, 4144, 2822), SolveHint(60.6, 36.2, SCALE, 4144, 2822)]
    ra, dec, r = catalog_cone(hints, 0.5)
    for h in hints:
        assert float(angular_sep_arcsec(h.ra, h.dec, ra, dec)) / 3600 + h.half_diagonal_deg + 0.5 <= r + 1e-9
    assert r == round(r, 1)
    ra, _, _ = catalog_cone([SolveHint(359.9, 0, SCALE, 100, 100), SolveHint(0.1, 0, SCALE, 100, 100)], 0.1)
    assert ra in (0.0, 360.0) or abs(ra) < 1e-6
    with pytest.raises(ValueError):
        catalog_cone([], 0.5)


# =========================================================================== segments and angles


def test_segments_after_meridian_flip():
    items = [("a", 1.0), ("b", 1.3), ("c", 181.2), ("d", 180.9), ("e", 0.8)]
    assert assign_segments(items, 20.0) == {"a": 1, "b": 1, "c": 2, "d": 2, "e": 3}
    assert assign_segments([], 20.0) == {}
    assert assign_segments([("a", 359.5), ("b", 0.5)], 20.0) == {"a": 1, "b": 1}


def test_angles():
    assert angle_diff_deg(359.0, 1.0) == pytest.approx(2.0)
    assert camera_angle_deg(180.6) == pytest.approx(0.6)
    assert circular_median_deg([359.0, 1.0, 0.5]) == pytest.approx(0.5)
    assert circular_median_deg([179.5, 0.5], period=180.0) in (pytest.approx(0.0), pytest.approx(180.0))
    assert math.isnan(circular_median_deg([]))


def test_saturated_bright_stars_do_not_break_the_vote():
    """OIII under a bright sky (sept23): the brightest stars saturate and are rejected by the detection;
    the vote must then reach further down the catalogue."""
    from dataclasses import replace as dc_replace
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=W, height=H, orientation_deg=180.6, parity=1)
    img, cat = synthetic_sky(truth, (H, W), n=1500, seed=5)
    x, y = truth.world_to_pixel(cat.ra, cat.dec)          # catalogue ≈ the field, as on the real 4° field
    keep = (x > -20) & (x < W + 20) & (y > -20) & (y < H + 20)
    cat = CatalogStars(cat.ra[keep], cat.dec[keep], cat.mag[keep])
    sat = float(np.percentile(img, 99.9))
    stars = detect_stars(img, saturation=sat)
    assert stars.n_saturated >= 20
    s = SolveSettings(sip_order=0, min_matches=15, vote_stars=30, vote_catalog=30, pointing_tolerance_deg=0.05)
    assert solve_field(stars, cat, SolveHint(60.44, 36.11, SCALE, W, H), s).ok
    blind = dc_replace(stars, n_saturated=0)      # without the correction (before the fix)
    assert not solve_field(blind, cat, SolveHint(60.44, 36.11, SCALE, W, H), s).ok
