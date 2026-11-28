"""Star detection on a calibrated frame (pure numpy/scipy, no file access).

Used by the plate solver (M3) and, later, by the image-quality step (FWHM, star counts).
The image is never modified: detection works on a background-subtracted, smoothed copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .blocks import fit_block
from .masks import block_median, upsample_nearest


@dataclass(frozen=True)
class Stars:
    """Detected stars, brightest first. Pixel coordinates are 0-based (x = column, y = row)."""

    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    noise_adu: float = 0.0       # robust noise of the background-subtracted image
    background_adu: float = 0.0  # median background
    n_saturated: int = 0         # stars rejected because saturated (the brightest of the field)

    @property
    def n(self) -> int:
        return int(self.x.size)

    def brightest(self, n: int) -> "Stars":
        return Stars(self.x[:n], self.y[:n], self.flux[:n], self.noise_adu, self.background_adu, self.n_saturated)


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def background_map(img: np.ndarray, block: int = 64) -> np.ndarray:
    """Smooth background: block medians (block size fitted to the image), upsampled."""
    b = fit_block(img.shape, block)
    small = block_median(img, b)
    if small.size > 1:
        small = ndimage.median_filter(small, size=3, mode="nearest")
    return upsample_nearest(small, b, img.shape)


def detect_stars(img: np.ndarray, *, threshold_sigma: float = 5.0, max_stars: int = 300,
                 background_block: int = 64, edge_px: int = 4, saturation: float | None = None,
                 max_peak_fraction: float = 0.7) -> Stars:
    """Local maxima of the smoothed, background-subtracted image above ``threshold_sigma``.

    - Centroid: intensity-weighted over a 7×7 window (sub-pixel).
    - Single hot pixels and cosmic rays are rejected: more than ``max_peak_fraction`` of the
      3×3 flux in one pixel.
    - Stars within ``edge_px`` of the border, and saturated stars when ``saturation`` is given,
      are dropped (their centroids are biased).
    """
    data = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, np.nanmedian(data))
    bg = background_map(data, background_block)
    resid = data - bg
    noise = _robust_sigma(resid[::2, ::2])
    smooth = ndimage.gaussian_filter(resid, 1.0)
    snoise = _robust_sigma(smooth[::2, ::2]) or noise
    if snoise <= 0:
        return Stars(np.empty(0), np.empty(0), np.empty(0), noise, float(np.median(bg)))
    peaks = (smooth == ndimage.maximum_filter(smooth, size=5)) & (smooth > threshold_sigma * snoise)
    h, w = data.shape
    peaks[:edge_px + 3, :] = peaks[h - edge_px - 3:, :] = False
    peaks[:, :edge_px + 3] = peaks[:, w - edge_px - 3:] = False
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-smooth[ys, xs])
    ys, xs = ys[order], xs[order]
    # Bounded work: a few times the requested number of candidates.
    ys, xs = ys[:max_stars * 4], xs[:max_stars * 4]
    out_x, out_y, out_f = [], [], []
    n_sat = 0
    r = 3
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    for y0, x0 in zip(ys, xs):
        win = resid[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        core = resid[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
        core_sum = float(core[core > 0].sum())
        if core_sum <= 0 or resid[y0, x0] / core_sum > max_peak_fraction:
            continue
        if saturation is not None and float(data[y0 - 1:y0 + 2, x0 - 1:x0 + 2].max()) >= saturation:
            n_sat += 1
            continue
        wpos = np.clip(win, 0, None)
        tot = float(wpos.sum())
        if tot <= 0:
            continue
        out_x.append(x0 + float((wpos * dx).sum()) / tot)
        out_y.append(y0 + float((wpos * dy).sum()) / tot)
        out_f.append(float(win.sum()))
        if len(out_x) >= max_stars:
            break
    x, y, f = np.array(out_x), np.array(out_y), np.array(out_f)
    order = np.argsort(-f)
    return Stars(x[order], y[order], f[order], noise, float(np.median(bg)), n_sat)
