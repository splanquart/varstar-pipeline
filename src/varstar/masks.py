"""Bad-pixel map (bit mask) and amp-glow mask. Pixels are never modified: measurements get flagged.

BPM bits:
    1  HOT        dark excess > hot_pixel_sigma x single-frame noise (warm, normally well corrected)
    2  VERY_HOT   dark excess > hot_pixel_severe_sigma x single-frame noise
    4  UNSTABLE   per-pixel temporal noise across darks > unstable_sigma_factor x median (RTS...)
    8  COLD       dark deficit > cold_pixel_sigma x single-frame noise
    16 DEAD       master flat < dead_pixel_flat_frac x local flat level
    32 SATURATED  master dark at the saturation level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HOT, VERY_HOT, UNSTABLE, COLD, DEAD, SATURATED = 1, 2, 4, 8, 16, 32
BIT_NAMES = {HOT: "hot", VERY_HOT: "very_hot", UNSTABLE: "unstable", COLD: "cold", DEAD: "dead",
             SATURATED: "saturated"}
BIT_NAMES_FR = {HOT: "tièdes", VERY_HOT: "très chauds", UNSTABLE: "instables", COLD: "froids",
                DEAD: "morts (flat)", SATURATED: "saturés (dark)"}


def block_median(a: np.ndarray, block: int) -> np.ndarray:
    """Median of non-overlapping blocks (edge blocks included, possibly smaller)."""
    h, w = a.shape
    nby, nbx = -(-h // block), -(-w // block)
    out = np.empty((nby, nbx), dtype=np.float32)
    for i in range(nby):
        rows = a[i * block:(i + 1) * block]
        for j in range(nbx):
            out[i, j] = np.median(rows[:, j * block:(j + 1) * block])
    return out


def upsample_nearest(small: np.ndarray, block: int, shape: tuple[int, int]) -> np.ndarray:
    big = np.repeat(np.repeat(small, block, axis=0), block, axis=1)
    return big[:shape[0], :shape[1]]


def robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


@dataclass
class MaskResult:
    mask: np.ndarray
    counts: dict[str, int]
    info: dict[str, Any]


def build_bpm(dark_adu: np.ndarray | None, sigma_single: float | None, unstable: np.ndarray | None,
              flat_adu: np.ndarray | None, cfg: dict[str, Any], saturation_adu: float = 65532) -> MaskResult:
    m = cfg["masks"]
    shape = (dark_adu if dark_adu is not None else flat_adu).shape
    bpm = np.zeros(shape, dtype=np.uint8)
    info: dict[str, Any] = {}
    if dark_adu is not None:
        med = float(np.median(dark_adu[::4, ::4]))
        sigma = sigma_single or robust_sigma(dark_adu[::4, ::4])
        info.update(dark_median_adu=med, noise_sigma_adu=sigma,
                    noise_source="bruit temporel d'une pose" if sigma_single else "dispersion du master")
        excess = dark_adu - med
        bpm[excess > float(m["hot_pixel_sigma"]) * sigma] |= HOT
        bpm[excess > float(m.get("hot_pixel_severe_sigma", 50.0)) * sigma] |= VERY_HOT
        bpm[excess < -float(m.get("cold_pixel_sigma", 5.0)) * sigma] |= COLD
        bpm[dark_adu >= 0.98 * saturation_adu] |= SATURATED
    if unstable is not None:
        bpm[unstable.astype(bool)] |= UNSTABLE
    if flat_adu is not None:
        block = int(m.get("flat_local_block_px", 32))
        local = upsample_nearest(block_median(flat_adu, block), block, flat_adu.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(local > 0, flat_adu / local, 0.0)
        bpm[rel < float(m["dead_pixel_flat_frac"])] |= DEAD
    counts = {BIT_NAMES[b]: int(np.count_nonzero(bpm & b)) for b in BIT_NAMES}
    counts["total"] = int(np.count_nonzero(bpm))
    return MaskResult(bpm, counts, info)


def build_amp_glow(dark_adu: np.ndarray, cfg: dict[str, Any]) -> MaskResult:
    """Zones where the smoothed master dark rises above its typical level (amp glow corners/edges)."""
    m = cfg["masks"]
    block = int(m.get("amp_glow_block_px", 64))
    k = float(m.get("amp_glow_sigma", 5.0))
    bm = block_median(dark_adu, block)
    level = float(np.median(bm))
    spread = max(robust_sigma(bm), 0.25)
    min_excess = float(m.get("amp_glow_min_adu", 2.0))
    threshold = level + max(k * spread, min_excess)
    small = (bm > threshold).astype(np.uint8)
    mask = upsample_nearest(small, block, dark_adu.shape).astype(np.uint8)
    frac = float(mask.mean())
    return MaskResult(mask, {"pixels": int(mask.sum())}, {
        "block_px": block, "level_adu": level, "block_sigma_adu": spread, "threshold_adu": threshold,
        "fraction": frac, "max_excess_adu": float(bm.max() - level)})
