"""Block and binning sizes fitted to the image size (https://github.com/splanquart/varstar-pipeline/issues/4)."""

from hypothesis import given
from hypothesis import strategies as st

from varstar.domain.blocks import MIN_BLOCKS, block_note, fit_bin, fit_block, shape_text


def test_fit_block_keeps_four_blocks_per_side():
    assert fit_block((2822, 4144), 64) == 64                # reference sensor: unchanged
    assert fit_block((64, 96), 64) == 16 and fit_block((64, 96), 32) == 16
    assert fit_block((64, 96), 8) == 8
    assert fit_block((3, 3), 16) == 1 and fit_block((0, 0), 16) == 1 and fit_block((5,), 16) == 1
    assert fit_block((64, 96), 64, min_blocks=2) == 32


def test_fit_bin_keeps_enough_binned_pixels():
    assert fit_bin((2822, 4144), 8, 64) == 8
    assert fit_bin((64, 96), 8, 64) == 1 and fit_bin((256, 512), 8, 64) == 4
    assert fit_bin((10, 10), 8, 64) == 1 and fit_bin((5,), 8, 64) == 1


@given(st.integers(0, 5000), st.integers(0, 5000), st.integers(1, 256))
def test_fitted_block_never_exceeds_the_request(h, w, block):
    used = fit_block((h, w), block)
    assert 1 <= used <= block
    if used > 1:
        assert h // used >= MIN_BLOCKS and w // used >= MIN_BLOCKS


def test_block_note():
    assert block_note("masque", "masks.x", 16, 16, (64, 96)) is None
    note = block_note("masque d'amp glow", "masks.amp_glow_block_px", 64, 16, (64, 96))
    assert note == ("masque d'amp glow : blocs de 16 px au lieu de 64 px (masks.amp_glow_block_px) pour garder "
                    "au moins 4 blocs par côté sur une image de 96×64 px")
    assert shape_text((64, 96)) == "96×64" and shape_text((7,)) == "7"
