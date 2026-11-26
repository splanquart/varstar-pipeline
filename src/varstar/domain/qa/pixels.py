"""Statistics computed on pixel arrays already read (quick stats, binned flats, calibrated frames).

The arrays come from the adapters (``adapters.pixel_stats``); nothing here opens a file.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from ..model import Frame
from ..units import detect_adu_scale


def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def clipped_mean(x: np.ndarray, med: float, sigma: float, k: float = 5.0) -> float:
    """Mean of the values within med ± k·sigma.

    Unlike the median, it is not quantised (the raw values are multiples of 4 ADU) and it ignores
    hot pixels and cosmic rays.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return float(np.mean(x))
    sel = x[np.abs(x - med) <= k * sigma]
    return float(np.mean(sel)) if sel.size else med


def frame_quick_stats(ints: np.ndarray, data: np.ndarray, step: int = 4,
                      saturation: float = 65532.0) -> dict[str, Any]:
    """Sub-sampled statistics of one raw frame (central region avoids the amp glow in the corners).

    ``ints``: the raw integer values (for the ADU scale); ``data``: the same frame in float32 ADU.
    """
    sub = data[::step, ::step]
    h, w = sub.shape
    cy, cx = max(h // 8, 1), max(w // 8, 1)
    center = sub[cy:h - cy, cx:w - cx]
    med = float(np.median(center))
    sig = robust_sigma(center)
    level = clipped_mean(center, med, sig)
    sel = center[np.abs(center - med) <= 5 * sig] if sig > 0 else center
    noise = float(np.std(sel))  # not quantised, unlike the MAD of multiples of 4 ADU
    corners = {
        "haut_gauche": sub[:cy, :cx], "haut_droit": sub[:cy, w - cx:],
        "bas_gauche": sub[h - cy:, :cx], "bas_droit": sub[h - cy:, w - cx:],
    }
    # 3x3 blocks of the central region: a light leak is rarely uniform.
    bh, bw = center.shape[0] // 3, center.shape[1] // 3
    blocks = [clipped_mean(b, float(np.median(b)), sig) for b in
              (center[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] for i in range(3) for j in range(3))]
    return {
        "median": level,
        "robust_sigma": noise,
        "p005": float(np.percentile(center, 0.5)),
        "corners": {k: clipped_mean(v, float(np.median(v)), sig) - level for k, v in corners.items()},
        "corner_npix": int(cy * cx),
        "block_range": max(blocks) - min(blocks),
        "frac_saturated": float(np.mean(sub >= saturation)),
        "adu_scale": detect_adu_scale(ints[::step, ::step]),
        "shape": list(data.shape),
    }


def bin_frame(data: np.ndarray, bin_px: int = 8) -> np.ndarray:
    """Frame averaged in bin_px x bin_px blocks (for flat comparisons); the remainder is cropped."""
    h, w = (data.shape[0] // bin_px) * bin_px, (data.shape[1] // bin_px) * bin_px
    return data[:h, :w].reshape(h // bin_px, bin_px, w // bin_px, bin_px).mean(axis=(1, 3)).astype(np.float32)


def flat_compare_frames(frames: Sequence[Frame], max_frames: int) -> list[Frame]:
    """At most ``max_frames`` flats of a series, evenly spread in time."""
    frames = sorted(frames, key=lambda f: f.date_obs)
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in sorted(set(idx))]
    return frames


def combine_quick_flat(binned: Sequence[np.ndarray], dark_level: float) -> np.ndarray:
    """Median of binned flats, minus the dark level, normalised to a median of 1."""
    flat = np.median(np.stack(binned), axis=0) - np.float32(dark_level)
    return (flat / np.median(flat)).astype(np.float32)


def block_median(img: np.ndarray, block: int) -> np.ndarray:
    h, w = (img.shape[0] // block) * block, (img.shape[1] // block) * block
    v = img[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3).reshape(
        h // block, w // block, -1)
    return np.median(v, axis=2)


FLAT_RATIO_BLOCK = 16       # binned px: scale of the vignetting (large-scale shape of the ratio)
FLAT_RATIO_MIN_BLOCKS = 4   # blocks per side, at least, for a meaningful large-scale shape
FLAT_RATIO_MIN_BLOCK = 3    # smaller blocks cannot tell dust shadows (several bins) from the shape


def flat_ratio_block(shape: tuple[int, ...], block: int = FLAT_RATIO_BLOCK) -> int:
    """Block size actually used by ``flat_ratio_metrics`` for binned flats of ``shape``.

    ``block`` shrinks so that each side holds at least ``FLAT_RATIO_MIN_BLOCKS`` blocks; 0 when
    even blocks of ``FLAT_RATIO_MIN_BLOCK`` do not fit (image too small to compare flats).
    See https://github.com/splanquart/varstar-pipeline/issues/2
    """
    if len(shape) != 2:
        return 0
    fitted = min(int(block), shape[0] // FLAT_RATIO_MIN_BLOCKS, shape[1] // FLAT_RATIO_MIN_BLOCKS)
    return fitted if fitted >= FLAT_RATIO_MIN_BLOCK else 0


def flat_ratio_metrics(a: np.ndarray, b: np.ndarray, block: int = FLAT_RATIO_BLOCK) -> dict[str, float] | None:
    """Compare two normalised (binned) flats; ``None`` when they are too small to be compared.

    shape_pct: peak-to-peak of the large-scale ratio (vignetting moved: rotation, refocus, panel);
    dust_pct: strongest local structure once the large scale is removed (dust shadows moved);
    block_px: block size used (smaller than ``block`` on small images, see ``flat_ratio_block``).
    """
    if a.shape != b.shape:
        raise ValueError(f"flats de tailles différentes : {a.shape} et {b.shape}")
    block = flat_ratio_block(a.shape, block)
    if not block:
        return None
    r = a / b
    r = r / np.median(r)
    coarse = block_median(r, block)
    shape = 100 * float(np.percentile(coarse, 99) - np.percentile(coarse, 1))
    up = np.kron(coarse, np.ones((block, block), dtype=np.float32))
    h, w = up.shape
    resid = r[:h, :w] / up - 1.0
    # 3x3 smoothing suppresses the pixel noise, dust shadows are several bins wide.
    small = block_median(resid, 3)
    noise = 100 * robust_sigma(resid)
    dust = 100 * float(np.max(np.abs(small)))
    return {"shape_pct": round(shape, 3), "dust_pct": round(dust, 3), "noise_pct": round(noise, 3),
            "block_px": block}


Zone = tuple[np.ndarray, np.ndarray]


def ampglow_zones(mask: np.ndarray, step: int = 4) -> list[Zone]:
    """Connected amp glow zones of a (sub-sampled) boolean mask, each with the band around it.

    Local comparison: the zone vs a 48-px band around it (a whole-field comparison would mostly
    measure the sky/nebula gradient). Zones or bands smaller than 16 pixels are ignored.
    """
    lab, nz = ndimage.label(mask)
    zones = []
    for zi in range(1, nz + 1):
        z = lab == zi
        band = ndimage.binary_dilation(z, iterations=max(1, 48 // step)) & ~mask
        if z.sum() >= 16 and band.sum() >= 16:
            zones.append((z, band))
    return zones


def calibrated_frame_stats(sub: np.ndarray, zones: Sequence[Zone]) -> dict[str, Any]:
    """Background, noise, invalid pixels and amp glow residuals of one (sub-sampled) calibrated frame."""
    finite = np.isfinite(sub)
    nan = float(1 - finite.mean())
    vals = sub[finite]
    med = float(np.median(vals))
    sig = robust_sigma(vals)
    clip = vals[np.abs(vals - med) < 3 * sig] if sig > 0 else vals
    bg = float(np.median(clip))
    zero = float(np.mean(vals == 0))
    neg = float(np.mean(vals < bg - 5 * sig)) if sig > 0 else 0.0
    resid = [float(np.median(sub[z & finite]) - np.median(sub[b & finite]))
             for z, b in zones if z.shape == sub.shape]
    return {"background_adu": round(bg, 2), "sigma_adu": round(sig, 2), "frac_nan": nan, "frac_zero": zero,
            "frac_low_5sigma": neg, "ampglow_zone_resid_adu": [round(r, 2) for r in resid]}
