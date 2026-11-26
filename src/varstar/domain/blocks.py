"""Block and binning sizes fitted to the image size.

The defaults (amp glow blocks of 64 px, local flat blocks of 32 px, flats binned x8 then 16-px
blocks for P4) suit the ASI294MM in bin 2 (4144x2822). A smaller image (other sensor, stronger
binning, ROI) would get one or two blocks only: the statistics between blocks would mean nothing,
silently. The sizes shrink to keep at least ``MIN_BLOCKS`` blocks per side, and every adaptation
is reported (``block_note``).
See https://github.com/splanquart/varstar-pipeline/issues/4
"""

from __future__ import annotations

from typing import Sequence

MIN_BLOCKS = 4   # blocks per side, at least


def fit_block(shape: Sequence[int], block: int, min_blocks: int = MIN_BLOCKS) -> int:
    """Largest block size <= ``block`` leaving at least ``min_blocks`` blocks per side (1 at least)."""
    if len(shape) < 2:
        return 1
    return max(1, min(int(block), int(shape[0]) // min_blocks, int(shape[1]) // min_blocks))


def fit_bin(shape: Sequence[int], bin_px: int, min_side: int) -> int:
    """Largest binning <= ``bin_px`` leaving at least ``min_side`` binned pixels per side (1 at least)."""
    if len(shape) < 2:
        return 1
    return max(1, min(int(bin_px), min(int(shape[0]), int(shape[1])) // min_side))


def shape_text(shape: Sequence[int]) -> str:
    """``(rows, cols)`` as « cols×rows » (width first, like the sensor sizes)."""
    return f"{shape[1]}×{shape[0]}" if len(shape) >= 2 else "×".join(str(n) for n in shape)


def block_note(what: str, key: str, requested: int, used: int, shape: Sequence[int]) -> str | None:
    """French note when a block size was reduced (``None`` when unchanged)."""
    if used >= requested:
        return None
    return (f"{what} : blocs de {used} px au lieu de {requested} px ({key}) pour garder au moins "
            f"{MIN_BLOCKS} blocs par côté sur une image de {shape_text(shape)} px")
