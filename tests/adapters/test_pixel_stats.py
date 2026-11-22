"""``PixelStatsCache`` on synthetic FITS files, and the raw checks end to end on real pixel noise."""

import numpy as np
import pytest
from astropy.io import fits

from fitsgen import noise, series, two_nights, write_frame
from varstar.adapters.pixel_stats import PixelStatsCache, binned_frame, quick_stats, read_image, read_raw
from varstar.adapters.workspace import Workspace
from varstar.application.check import raw_checks
from varstar.inventory import build_inventory


@pytest.fixture(autouse=True)
def full_pixels(cfg):
    cfg["qa"]["pixel_step"] = 1  # small synthetic frames: use every pixel
    cfg["runtime"]["processes"] = 1


def checks_of(root, cfg, workdir):
    inv = build_inventory(root, cfg)
    return raw_checks(inv, cfg, store=Workspace(workdir), pixels=PixelStatsCache.from_config(workdir, cfg))


def test_read_raw_handles_bzero(tmp_path):
    p = write_frame(tmp_path / "a.fits", "DARK", 300, "2021-09-07T02:00:00", data=noise(516, 12, 1))
    assert fits.getheader(p)["BZERO"] == 32768
    ints, data = read_raw(p)
    assert ints.dtype == np.int64 and data.dtype == np.float32 and (ints % 4 == 0).all()
    assert quick_stats(str(p), 1)["adu_scale"] == 4
    assert binned_frame(str(p), 8).shape == (16, 20)
    assert read_image(p).dtype == np.float32


def test_offset_jump_detected(tmp_path, cfg):
    items, tables = checks_of(two_nights(tmp_path / "Y", level2=541.0), cfg, tmp_path / "work")
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
    items, _ = checks_of(two_nights(tmp_path / "Y", leak=True), cfg, tmp_path / "work")
    assert [i.check for i in items if i.check in ("P1", "P2")] == ["P2"]


def test_dark_above_lights_is_error(tmp_path, cfg):
    root = tmp_path / "Y"
    series(root, "light/H/SESSION_s", "LIGHT", 300, "2021-09-06T23:00:00", 3, step_s=600, filt="H",
           shape=(128, 160), make=lambda i: noise(520, 12, i))
    series(root, "dark/x", "DARK", 300, "2021-09-07T02:00:00", 4, step_s=310, prefix="", shape=(128, 160),
           make=lambda i: noise(545, 12, 50 + i))
    items, _ = checks_of(root, cfg, tmp_path / "work")
    p8 = [i for i in items if i.check == "P8"]
    assert p8 and p8[0].severity == "error"


def test_frame_stats_are_cached_by_size_and_mtime(tmp_path, cfg):
    root = two_nights(tmp_path / "Y")
    inv = build_inventory(root, cfg)
    frames = inv.calibration_sets["DARK"][0].frames
    cache = PixelStatsCache(tmp_path / "work", step=1)
    first = cache.frame_stats(frames, 65532)
    assert (tmp_path / "work" / "qa" / "frame_stats.json").exists()
    again = PixelStatsCache(tmp_path / "work", step=1)
    assert again.data and again.frame_stats(frames, 65532) == first
    assert PixelStatsCache(tmp_path / "work", step=2).data == {}           # another sub-sampling: not reused
    (tmp_path / "work" / "qa" / "frame_stats.json").write_text("{broken")
    assert PixelStatsCache(tmp_path / "work", step=1).data == {}


def test_quick_flat_is_normalised_and_cached(tmp_path, cfg):
    paths = series(tmp_path / "Y", "flat/s", "FLAT", 1.13, "2021-09-07T13:21:00", 3, step_s=2, filt="H",
                   shape=(64, 96), make=lambda i: noise(30000, 100, i, shape=(64, 96)))
    inv = build_inventory(tmp_path / "Y", cfg)
    frames = inv.calibration_sets["FLAT"][0].frames
    cache = PixelStatsCache(tmp_path / "work", step=1)
    flat = cache.quick_flat("FLAT-H", frames, 8, 516.0)
    assert flat.shape == (8, 12) and flat.dtype == np.float32 and np.median(flat) == pytest.approx(1)
    assert len(list((tmp_path / "work" / "qa" / "flats").glob("*.npy"))) == 1
    assert np.array_equal(cache.quick_flat("FLAT-H", frames, 8, 516.0), flat)
    assert all(p.exists() for p in paths)


def test_worker_processes_give_the_same_results(tmp_path, cfg):
    root = two_nights(tmp_path / "Y")
    inv = build_inventory(root, cfg)
    frames = [f for s in inv.calibration_sets["DARK"] for f in s.frames]            # 10 frames > 4: pool used
    serial = PixelStatsCache(tmp_path / "w1", step=1).frame_stats(frames, 65532)
    pooled = PixelStatsCache(tmp_path / "w2", step=1, processes=2).frame_stats(frames, 65532)
    assert pooled == serial
    flats = frames[:3]
    assert np.array_equal(PixelStatsCache(tmp_path / "w2", step=1, processes=2).quick_flat("x", flats, 8, 0.0),
                          PixelStatsCache(tmp_path / "w1", step=1).quick_flat("x", flats, 8, 0.0))
