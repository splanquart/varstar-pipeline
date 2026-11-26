"""``SirilMasterBuilder`` parts that need no Siril: statistics of the source frames read from FITS."""

import numpy as np

from fitsgen import write_frame
from varstar.adapters.fits_io import read_frame
from varstar.adapters.siril_masters import frame_stats


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
    assert frame_stats(frames[:3], temporal=False)[1] == {}
