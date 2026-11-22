import struct

import numpy as np
import pytest

from varstar.config import load_config
from varstar.imageio import read_siril_output, read_xisf
from varstar.masks import COLD, DEAD, HOT, SATURATED, UNSTABLE, VERY_HOT, build_amp_glow, build_bpm, block_median
from varstar.masters import choose_rejection, frame_stats
from varstar.fitsinfo import read_frame

from fitsgen import write_frame


@pytest.mark.parametrize("n,code", [(3, "p"), (7, "p"), (8, "s"), (14, "s"), (15, "w"), (25, "w"), (26, "l"),
                                    (65, "l")])
def test_choose_rejection(n, code):
    rej = choose_rejection(n, load_config())
    assert rej.code == code
    assert rej.siril_args().startswith(f"rej {code} ")


def test_bpm_categories():
    cfg = load_config()
    rng = np.random.default_rng(0)
    dark = (500 + rng.normal(0, 1, (64, 64))).astype(np.float32)
    dark[1, 1] = 500 + 10 * 8        # hot (10 sigma, sigma=8)
    dark[2, 2] = 500 + 100 * 8       # very hot
    dark[3, 3] = 500 - 10 * 8        # cold
    dark[4, 4] = 65532               # saturated
    unstable = np.zeros((64, 64), np.uint8)
    unstable[5, 5] = 1
    flat = np.full((64, 64), 30000, np.float32)
    flat[6, 6] = 3000                # dead
    res = build_bpm(dark, 8.0, unstable, flat, cfg)
    m = res.mask
    assert m[1, 1] == HOT
    assert m[2, 2] == HOT | VERY_HOT
    assert m[3, 3] == COLD
    assert m[4, 4] & SATURATED
    assert m[5, 5] == UNSTABLE
    assert m[6, 6] == DEAD
    assert m[10, 10] == 0
    assert res.counts["dead"] == 1 and res.counts["cold"] == 1


def test_bpm_flat_vignetting_is_not_dead():
    cfg = load_config()
    yy, xx = np.mgrid[0:256, 0:256]
    flat = (30000 * (1 - 0.5 * ((xx - 128) ** 2 + (yy - 128) ** 2) / 128 ** 2 / 2)).astype(np.float32)
    assert build_bpm(None, None, None, flat, cfg).counts["dead"] == 0


def test_amp_glow_corner():
    cfg = load_config()
    rng = np.random.default_rng(1)
    dark = (520 + rng.normal(0, 3, (512, 768))).astype(np.float32)
    dark[:128, -128:] += 25          # glowing corner
    res = build_amp_glow(dark, cfg)
    assert res.mask[:128, -128:].all()
    assert res.mask[200:, :600].sum() == 0
    assert 0.03 < res.info["fraction"] < 0.06


def test_block_median_edges():
    a = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    bm = block_median(a, 4)
    assert bm.shape == (3, 2)
    assert bm[-1, -1] == np.median(a[8:, 4:])


def test_frame_stats_unstable_ignores_single_cosmic(tmp_path):
    rng = np.random.default_rng(2)
    frames = []
    for i in range(10):
        d = 500 + rng.normal(0, 8, (40, 40))
        d[10, 10] = 500 + (300 if i % 2 else -300)    # RTS-like pixel, unstable
        if i == 3:
            d[20, 20] = 5000                            # single cosmic ray
        data = (np.round(d / 4) * 4).astype(np.uint16)
        p = write_frame(tmp_path / f"d{i}.fits", "DARK", 300, f"2021-09-07T02:{i:02d}:00", data=data)
        frames.append(read_frame(p, tmp_path))
    stats, extra = frame_stats(frames, temporal=True)
    assert len(stats) == 10 and abs(stats[0]["median"] - 500) < 5
    assert 6 < extra["sigma_single"] < 10
    pstd = extra["pixel_std"]
    med = np.median(pstd)
    assert pstd[10, 10] > 5 * med
    assert pstd[20, 20] < 3 * med


def _write_xisf(path, data):
    data = data.astype("<f4")
    h, w = data.shape
    offset = 4096
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">'
           f'<Image geometry="{w}:{h}:1" sampleFormat="Float32" bounds="0:1" colorSpace="Gray" '
           f'location="attachment:{offset}:{data.nbytes}"/></xisf>').encode()
    with open(path, "wb") as fh:
        fh.write(b"XISF0100" + struct.pack("<I", len(xml)) + b"\0" * 4 + xml)
        fh.write(b"\0" * (offset - fh.tell()))
        fh.write(data.tobytes())


def test_read_xisf(tmp_path):
    data = np.random.default_rng(3).random((30, 50)).astype(np.float32)
    _write_xisf(tmp_path / "m.xisf", data)
    np.testing.assert_array_equal(read_xisf(tmp_path / "m.xisf"), data)


def test_read_siril_output_rejects_unnormalised(tmp_path):
    from astropy.io import fits
    fits.PrimaryHDU(np.full((4, 4), 0.01, np.float32)).writeto(tmp_path / "ok.fits")
    fits.PrimaryHDU(np.full((4, 4), 500, np.float32)).writeto(tmp_path / "bad.fits")
    data, _ = read_siril_output(tmp_path / "ok.fits")
    assert np.allclose(data, 0.01 * 65535)
    with pytest.raises(ValueError):
        read_siril_output(tmp_path / "bad.fits")
