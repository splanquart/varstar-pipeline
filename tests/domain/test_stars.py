"""Star detection on calibrated frames."""

import numpy as np

from varstar.domain.stars import background_map, detect_stars


def frame(stars, shape=(120, 160), sky=500.0, noise=5.0, seed=0, gradient=0.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    img = sky + gradient * xx + rng.normal(0, noise, shape)
    for x, y, flux in stars:
        img += flux / (2 * np.pi * 1.5 ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))
    return img.astype(np.float32)


def test_detects_and_centroids_brightest_first():
    truth = [(40.3, 30.7, 20000.0), (100.6, 80.2, 8000.0), (130.1, 20.4, 3000.0)]
    st = detect_stars(frame(truth))
    assert st.n == 3
    assert np.allclose(st.x, [t[0] for t in truth], atol=0.15)
    assert np.allclose(st.y, [t[1] for t in truth], atol=0.15)
    assert list(np.argsort(-st.flux)) == [0, 1, 2]
    assert abs(st.background_adu - 500) < 2 and 4 < st.noise_adu < 6
    assert st.brightest(2).n == 2


def test_rejects_hot_pixels_edges_and_saturated_stars():
    img = frame([(40.0, 30.0, 20000.0), (3.0, 60.0, 20000.0), (100.0, 80.0, 400000.0)])
    img[90, 20] += 5000.0          # single hot pixel / cosmic ray
    st = detect_stars(img, saturation=6000.0)
    assert st.n == 1 and abs(st.x[0] - 40) < 0.2
    assert detect_stars(img).n == 2   # without saturation limit, the bright star is kept


def test_gradient_and_nan_and_blank():
    st = detect_stars(frame([(80.0, 60.0, 10000.0)], gradient=2.0))
    assert st.n == 1
    img = frame([(80.0, 60.0, 10000.0)])
    img[5:10, 5:10] = np.nan
    assert detect_stars(img).n == 1
    assert detect_stars(np.full((50, 50), 100.0, np.float32)).n == 0
    assert background_map(np.ones((40, 40), np.float32)).shape == (40, 40)


def test_max_stars():
    stars = [(20.0 + 25 * i, 20.0 + 18 * j, 5000.0 + 100 * i + 10 * j) for i in range(5) for j in range(5)]
    st = detect_stars(frame(stars, shape=(130, 150)), max_stars=7)
    assert st.n == 7 and st.flux[0] >= st.flux[-1]
