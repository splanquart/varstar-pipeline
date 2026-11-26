"""« minisky » : a tiny, deterministic N.I.N.A.-like dataset with planted defects.

Three nights on a 64×96 px sensor (≈12 kB per frame, ≈45 frames):

- night ``s1`` (2021-09-06): lights H and S, darks of the same night, flats the next
  morning (one H flat misfiled in a ``dark/`` folder), dark-flats → level A;
- night ``s2`` (2021-09-09): lights H, flats the next morning, but darks taken with the
  cooler at −5 °C instead of −12 °C: they are rejected (ΔT > tolerance) and the darks of the
  nearest night (s1) are borrowed → level B;
- night ``s3`` (2021-09-20): lights S and darks, no flat within 7 days → level C.

Pixel model (ADU before ×4 quantisation): offset, dark current, read noise, a hot pixel,
an amp glow corner, a dead pixel (low response), a vignetted flat field, a few Gaussian
stars and a light-only hot pixel (cosmic ray) that must survive calibration untouched.
File names follow N.I.N.A. (spaces in ``NGC 1499``). Every file gets a fixed mtime.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from fitsgen import series

SHAPE = (64, 96)
OFFSET = 500.0
DARK_CURRENT = 0.02          # ADU/s at −12 °C
DARK_CURRENT_WARM = 0.06     # ADU/s at −5 °C
READ_NOISE = 3.0
GAIN_E_PER_ADU16 = 0.25      # EGAIN ≈ 1 e⁻/ADU14, samples stored ×4
HOT = (5, 7)                 # +10 ADU/s in every frame
HOT_RATE = 10.0
AMP_GLOW = (slice(0, 16), slice(80, 96))   # top-right corner
AMP_GLOW_RATE = 0.1          # ADU/s → +30 ADU in 300 s
DEAD = (40, 60)              # pixel response 30 %
DEAD_RESPONSE = 0.3
COSMIC = (20, 30)            # first H light of s1 only
COSMIC_ADU = 2000.0
SKY = 1000.0                 # ADU in 300 s at the optical centre
FLAT_LEVEL = 25000.0         # ≈38 % of the dynamic range
FLAT_NOISE = 100.0
STARS = [((32.0, 48.0), 40000.0), ((15.0, 20.0), 20000.0), ((50.0, 80.0), 30000.0)]
STAR_SIGMA = 1.5
LIGHT_EXP = 300.0
FLAT_EXP = {"H": 1.13, "S": 2.62}

# Configuration adapted to the image size (blocks of 64 px make no sense on 64×96).
CONFIG = {
    "masks": {"flat_local_block_px": 16, "amp_glow_block_px": 16},
    "qa": {"flat_compare_bin": 2},   # binned ×8, 64 px leave 8 px: too small to compare the flats (P4)
    "runtime": {"processes": 1},
    "paths": {"pixinsight_masters": None},
}


def vignetting() -> np.ndarray:
    y, x = np.indices(SHAPE, dtype=float)
    r2 = ((y - SHAPE[0] / 2) / SHAPE[0]) ** 2 + ((x - SHAPE[1] / 2) / SHAPE[1]) ** 2
    return 1.0 - 0.8 * r2          # 1 at the centre, ≈0.8 in the corners


def response() -> np.ndarray:
    r = vignetting()
    r[DEAD] *= DEAD_RESPONSE
    return r


def stars() -> np.ndarray:
    y, x = np.indices(SHAPE, dtype=float)
    img = np.zeros(SHAPE)
    for (yc, xc), flux in STARS:
        img += flux / (2 * np.pi * STAR_SIGMA ** 2) * np.exp(
            -((y - yc) ** 2 + (x - xc) ** 2) / (2 * STAR_SIGMA ** 2))
    return img


def q4(x: np.ndarray) -> np.ndarray:
    """14-bit sample stored ×4 in uint16, as N.I.N.A. writes the ASI294MM frames."""
    return (np.round(np.clip(x, 0, 65532) / 4) * 4).astype(np.uint16)


def _thermal(exp: float, rate: float) -> np.ndarray:
    d = np.full(SHAPE, OFFSET + rate * exp)
    d[HOT] += HOT_RATE * exp
    d[AMP_GLOW] += AMP_GLOW_RATE * exp
    return d


def thermal_noise(exp: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Read noise + shot noise of the uniform dark current (variance in ADU16² = signal / gain).

    QA check P1 tells an offset jump (noise unchanged) from an extra signal (noise grows)
    with exactly this shot noise. The hot pixel is kept noise-free so that its correction by
    the dark can be asserted sharply.
    """
    return rng.normal(0, np.sqrt(READ_NOISE ** 2 + rate * exp / GAIN_E_PER_ADU16), SHAPE)


def dark_maker(exp: float, rng: np.random.Generator, rate: float = DARK_CURRENT) -> Callable[[int], np.ndarray]:
    return lambda i: q4(_thermal(exp, rate) + thermal_noise(exp, rate, rng))


def flat_maker(exp: float, rng: np.random.Generator) -> Callable[[int], np.ndarray]:
    return lambda i: q4(_thermal(exp, DARK_CURRENT) + FLAT_LEVEL * response() + rng.normal(0, FLAT_NOISE, SHAPE))


def light_maker(rng: np.random.Generator, cosmic: bool = False) -> Callable[[int], np.ndarray]:
    def make(i: int) -> np.ndarray:
        d = _thermal(LIGHT_EXP, DARK_CURRENT) + (SKY + stars()) * response() + thermal_noise(
            LIGHT_EXP, DARK_CURRENT, rng)
        if cosmic and i == 0:
            d[COSMIC] += COSMIC_ADU
        return q4(d)
    return make


def _fix_mtimes(root: Path) -> None:
    """Deterministic mtimes (master ids and caches depend on them)."""
    base = dt.datetime(2021, 10, 1).timestamp()
    for i, p in enumerate(sorted(root.rglob("*.fits"))):
        os.utime(p, (base + i, base + i))


def make_minisky(root: Path) -> Path:
    """Write the minisky dataset under ``root`` (returned)."""
    rng = np.random.default_rng(20210906)
    kw = {"shape": SHAPE}
    cold = {"temp": -5.0, "extra": {"SET-TEMP": -5.0}}

    # Night s1 — level A.
    series(root, "light/H/SESSION_s1", "LIGHT", LIGHT_EXP, "2021-09-06T23:00:00", 3, step_s=400, filt="H",
           make=light_maker(rng, cosmic=True), **kw)
    series(root, "light/S/SESSION_s1", "LIGHT", LIGHT_EXP, "2021-09-07T00:30:00", 3, step_s=400, filt="S",
           make=light_maker(rng), **kw)
    series(root, "dark/darklight/300s", "DARK", LIGHT_EXP, "2021-09-07T02:00:00", 8, step_s=310, prefix="",
           make=dark_maker(LIGHT_EXP, rng), **kw)
    series(root, "flat/SESSION_s1", "FLAT", FLAT_EXP["H"], "2021-09-07T06:00:00", 4, step_s=2, filt="H",
           make=flat_maker(FLAT_EXP["H"], rng), **kw)
    # Trap: one H flat of the same series misfiled in a dark folder (the header wins).
    series(root, "dark/oops", "FLAT", FLAT_EXP["H"], "2021-09-07T06:00:08", 1, step_s=2, filt="H",
           make=flat_maker(FLAT_EXP["H"], rng), **kw)
    series(root, "flat/SESSION_s1", "FLAT", FLAT_EXP["S"], "2021-09-07T06:05:00", 5, step_s=3, filt="S",
           make=flat_maker(FLAT_EXP["S"], rng), **kw)
    series(root, "dark/darkflat", "DARKFLAT", FLAT_EXP["H"], "2021-09-07T07:00:00", 5, step_s=2, prefix="",
           make=dark_maker(FLAT_EXP["H"], rng), **kw)
    series(root, "dark/darkflat", "DARKFLAT", FLAT_EXP["S"], "2021-09-07T07:01:00", 5, step_s=3, prefix="",
           make=dark_maker(FLAT_EXP["S"], rng), **kw)

    # Night s2 — level B: darks of the night at −5 °C (rejected), s1 darks borrowed.
    series(root, "light/H/SESSION_s2", "LIGHT", LIGHT_EXP, "2021-09-09T23:00:00", 3, step_s=400, filt="H",
           make=light_maker(rng), **kw)
    series(root, "flat/SESSION_s2", "FLAT", FLAT_EXP["H"], "2021-09-10T06:00:00", 5, step_s=2, filt="H",
           make=flat_maker(FLAT_EXP["H"], rng), **kw)
    series(root, "dark/darklight/300s", "DARK", LIGHT_EXP, "2021-09-10T02:00:00", 5, step_s=310, prefix="",
           make=dark_maker(LIGHT_EXP, rng, DARK_CURRENT_WARM), **kw, **cold)

    # Night s3 — level C: no flat within 7 days.
    series(root, "light/S/SESSION_s3", "LIGHT", LIGHT_EXP, "2021-09-20T23:00:00", 3, step_s=400, filt="S",
           make=light_maker(rng), **kw)
    series(root, "dark/darklight/300s", "DARK", LIGHT_EXP, "2021-09-21T02:00:00", 5, step_s=310, prefix="",
           make=dark_maker(LIGHT_EXP, rng), **kw)

    _fix_mtimes(root)
    return root


def write_config(path: Path, library: Path) -> Path:
    cfg = {**CONFIG, "paths": {**CONFIG["paths"], "library": str(library)}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def file_state(root: Path) -> list[tuple[str, int, str]]:
    """(path, mtime_ns, sha256) of every file: raw frames must stay untouched."""
    import hashlib
    return sorted((str(p), p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in root.rglob("*") if p.is_file())
