"""Statistics on pixel arrays (no file): quick stats, binned flats, flat ratios, calibrated frames."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from builders import make_frames
from varstar.domain.model import FLAT
from varstar.domain.qa.pixels import (ampglow_zones, bin_frame, calibrated_frame_stats, clipped_mean,
                                      combine_quick_flat, flat_compare_frames, flat_ratio_block,
                                      flat_ratio_metrics,
                                      frame_quick_stats, robust_sigma)


def raw_frame(level=500, sigma=12, shape=(64, 96), scale=4, seed=0):
    rng = np.random.default_rng(seed)
    ints = (np.round(rng.normal(level / scale, sigma / scale, shape)) * scale).astype(np.int64)
    return ints, ints.astype(np.float32)


def test_robust_sigma_and_clipped_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 10, 20000)
    x[:50] = 1e6                                   # hot pixels / cosmic rays
    assert robust_sigma(x) == pytest.approx(10, rel=0.05)
    assert clipped_mean(x, float(np.median(x)), robust_sigma(x)) == pytest.approx(100, abs=0.3)
    assert np.isnan(robust_sigma(np.array([np.nan])))
    assert clipped_mean(np.array([1.0, 3.0]), 2.0, 0.0) == 2.0


def test_quick_stats_level_noise_and_scale():
    ints, data = raw_frame(level=500, sigma=12)
    s = frame_quick_stats(ints, data, step=1)
    assert s["median"] == pytest.approx(500, abs=1) and s["robust_sigma"] == pytest.approx(12, rel=0.1)
    assert s["adu_scale"] == 4 and s["shape"] == [64, 96] and s["frac_saturated"] == 0.0
    assert set(s["corners"]) == {"haut_gauche", "haut_droit", "bas_gauche", "bas_droit"}
    assert all(abs(v) < 3 for v in s["corners"].values()) and s["corner_npix"] == 8 * 12


def test_quick_stats_sees_a_bright_corner_and_saturation():
    ints, data = raw_frame()
    ints[:8, :12] += 40
    ints[30, 40] = 65532
    s = frame_quick_stats(ints, ints.astype(np.float32), step=1, saturation=65532)
    assert s["corners"]["haut_gauche"] == pytest.approx(40, abs=2)
    assert s["frac_saturated"] == pytest.approx(1 / (64 * 96))


@given(st.integers(1, 4), st.integers(1, 30), st.integers(1, 30), st.floats(-1e4, 1e4, allow_nan=False))
def test_bin_frame_shape_and_constant(bin_px, h, w, value):
    out = bin_frame(np.full((h, w), value, dtype=np.float32), bin_px)
    assert out.shape == (h // bin_px, w // bin_px) and out.dtype == np.float32
    assert np.allclose(out, np.float32(value), rtol=1e-6)


@given(arrays(np.float32, (16, 24), elements=st.floats(0, 1000, width=32)))
def test_bin_frame_preserves_the_mean(data):
    assert float(bin_frame(data, 4).mean()) == pytest.approx(float(data.mean()), rel=1e-4, abs=1e-3)


def test_combine_quick_flat_is_normalised():
    a = np.full((4, 4), 1100, np.float32)
    b = np.full((4, 4), 1300, np.float32)
    flat = combine_quick_flat([a, b, a], dark_level=100)
    assert flat.dtype == np.float32 and np.allclose(flat, 1)


def test_flat_compare_frames_spreads_in_time():
    frames = make_frames(10, FLAT, step_s=2)[::-1]
    picked = flat_compare_frames(frames, 4)
    assert [f.date_obs for f in picked] == sorted(f.date_obs for f in picked)
    assert picked[0].date_obs == min(f.date_obs for f in frames)
    assert picked[-1].date_obs == max(f.date_obs for f in frames) and len(picked) == 4
    assert len(flat_compare_frames(frames, 20)) == 10


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
    same = flat_ratio_metrics(a, b)
    moved = flat_ratio_metrics(a, c)
    assert same["dust_pct"] < 0.5 and same["shape_pct"] < 0.5
    assert moved["dust_pct"] > 2.0
    tilted = (a * (1 + 0.05 * xx / 300)).astype(np.float32)
    assert flat_ratio_metrics(a, tilted)["shape_pct"] > 2.0


def test_flat_ratio_block_shrinks_on_small_images():
    # https://github.com/splanquart/varstar-pipeline/issues/2
    assert flat_ratio_block((352, 518)) == 16            # ASI294MM bin 2, flats binned x8: unchanged
    assert flat_ratio_block((32, 48)) == 8              # at least 4 blocks per side
    assert flat_ratio_block((12, 400)) == 3
    assert flat_ratio_block((8, 12)) == 0               # 64x96 binned x8: too small
    assert flat_ratio_block((0, 0)) == 0 and flat_ratio_block((5,)) == 0


def test_flat_ratio_metrics_on_small_images():
    rng = np.random.default_rng(2)
    a, b = (1 + 0.001 * rng.standard_normal((2, 32, 48))).astype(np.float32)
    met = flat_ratio_metrics(a, b)
    assert met["block_px"] == 8 and met["shape_pct"] < 1 and met["dust_pct"] < 1
    assert flat_ratio_metrics(a[:8, :12], b[:8, :12]) is None
    assert flat_ratio_metrics(np.ones((0, 0), np.float32), np.ones((0, 0), np.float32)) is None
    with pytest.raises(ValueError, match="tailles différentes"):
        flat_ratio_metrics(a, b[:16])


@given(st.integers(0, 70), st.integers(0, 70))
def test_flat_ratio_metrics_never_raises_on_any_shape(h, w):
    a = np.ones((h, w), np.float32)
    met = flat_ratio_metrics(a, a)
    if min(h, w) < 12:
        assert met is None
    else:
        assert met["shape_pct"] == 0 and met["dust_pct"] == 0 and 3 <= met["block_px"] <= 16


def test_ampglow_zones_and_their_band():
    mask = np.zeros((40, 60), bool)
    mask[:6, 50:] = True          # one corner zone
    mask[20, 20] = True           # too small: ignored
    zones = ampglow_zones(mask, step=4)
    assert len(zones) == 1
    zone, band = zones[0]
    assert zone.sum() == 60 and not (band & mask).any() and band.sum() >= 16


def test_calibrated_frame_stats():
    rng = np.random.default_rng(3)
    sub = rng.normal(50, 5, (40, 60)).astype(np.float32)
    mask = np.zeros_like(sub, bool)
    mask[:6, 50:] = True
    sub[mask] -= 20
    sub[10, 10] = np.nan
    s = calibrated_frame_stats(sub, ampglow_zones(mask, 4))
    assert s["background_adu"] == pytest.approx(50, abs=1) and s["sigma_adu"] == pytest.approx(5, rel=0.2)
    assert s["frac_nan"] == pytest.approx(1 / sub.size) and s["frac_zero"] == 0
    assert s["ampglow_zone_resid_adu"][0] == pytest.approx(-20, abs=2)
    # A zone of another shape (different sub-sampling) is ignored.
    assert calibrated_frame_stats(sub[:20], ampglow_zones(mask, 4))["ampglow_zone_resid_adu"] == []
