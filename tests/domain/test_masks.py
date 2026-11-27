"""Bad-pixel map and amp-glow mask (pure numpy). Pixels are never modified: only flagged."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from builders import make_settings
from varstar.domain.masks import (BIT_NAMES, COLD, DEAD, HOT, SATURATED, UNSTABLE, VERY_HOT, block_median,
                                  build_amp_glow, build_bpm, upsample_nearest)


def load_config():
    return make_settings().masks


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


def test_block_sizes_shrink_on_small_images_and_say_so():
    # https://github.com/splanquart/varstar-pipeline/issues/4: one 64-px block on 64x96 px found nothing.
    cfg = load_config()
    rng = np.random.default_rng(2)
    dark = (520 + rng.normal(0, 3, (64, 96))).astype(np.float32)
    dark[:16, 80:] += 30             # glowing corner
    res = build_amp_glow(dark, cfg)
    assert res.info["block_px"] == 16 and res.mask[:16, 80:].all() and res.mask[20:, :80].sum() == 0
    (note,) = res.info["notes"]
    assert "blocs de 16 px au lieu de 64 px (masks.amp_glow_block_px)" in note and "96×64 px" in note
    flat = np.full((64, 96), 30000, np.float32)
    bpm = build_bpm(None, None, None, flat, cfg)
    assert bpm.info["flat_local_block_px"] == 16
    assert "blocs de 16 px au lieu de 32 px (masks.flat_local_block_px)" in bpm.info["notes"][0]


def test_no_note_at_the_reference_size():
    cfg = load_config()
    dark = np.full((512, 768), 520, np.float32)
    assert "notes" not in build_amp_glow(dark, cfg).info
    assert "notes" not in build_bpm(None, None, None, dark, cfg).info


def test_block_median_edges():
    a = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    bm = block_median(a, 4)
    assert bm.shape == (3, 2)
    assert bm[-1, -1] == np.median(a[8:, 4:])


# --------------------------------------------------------------------------- properties

ALL_BITS = sum(BIT_NAMES)
shapes = st.tuples(st.integers(4, 40), st.integers(4, 40))


@st.composite
def calibration_inputs(draw):
    shape = draw(shapes)
    dark = draw(arrays(np.float32, shape, elements=st.floats(400, 700, width=32)))
    flat = draw(arrays(np.float32, shape, elements=st.floats(1000, 40000, width=32)))
    unstable = draw(arrays(np.uint8, shape, elements=st.integers(0, 1)))
    return dark, flat, unstable


@given(calibration_inputs(), st.floats(0.5, 20))
def test_bpm_never_modifies_inputs_and_uses_known_bits(inputs, sigma):
    dark, flat, unstable = inputs
    before = [a.copy() for a in inputs]
    res = build_bpm(dark, sigma, unstable, flat, make_settings(masks={"flat_local_block_px": 4}).masks)
    assert all(np.array_equal(a, b) for a, b in zip(inputs, before))
    assert res.mask.shape == dark.shape
    assert not (res.mask.astype(int) & ~ALL_BITS).any()


@given(arrays(np.float32, shapes, elements=st.floats(400, 700, width=32)))
def test_amp_glow_mask_is_binary_and_input_untouched(dark):
    before = dark.copy()
    res = build_amp_glow(dark, make_settings(masks={"amp_glow_block_px": 4}).masks)
    assert np.array_equal(dark, before)
    assert res.mask.shape == dark.shape and set(np.unique(res.mask)) <= {0, 1}


@given(shapes, st.integers(1, 9), st.floats(-1e4, 1e4, allow_nan=False))
def test_block_median_of_constant_image_is_constant(shape, block, value):
    img = np.full(shape, value, np.float32)
    bm = block_median(img, block)
    assert bm.shape == (-(-shape[0] // block), -(-shape[1] // block))
    assert np.all(bm == np.float32(value))
    assert upsample_nearest(bm, block, shape).shape == shape
