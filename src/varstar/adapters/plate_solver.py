"""``PlateSolver`` adapter: the internal solver (``domain.astrometry``) on calibrated FITS files.

The images are read-only: only their pixels are used, the WCS is returned to the use case.
Series (light groups) are cut into chunks solved in parallel processes; within a chunk each
solution guides the next image.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

from ..domain.astrometry import ASTROMETRY_VERSION, CatalogStars, SolveJob, SolveResult, SolveSettings, solve_series
from .pixel_stats import read_image


def _solve_chunk(jobs: list[SolveJob], catalog: CatalogStars, settings: SolveSettings) -> dict[str, SolveResult]:
    """Top-level (runs in worker processes)."""
    return solve_series(jobs, read_image, catalog, settings)


class InternalPlateSolver:
    def __init__(self, processes: int = 1):
        self.processes = max(1, int(processes))

    def name(self) -> str:
        return f"interne v{ASTROMETRY_VERSION}"

    def _chunks(self, series: list[list[SolveJob]]) -> list[list[SolveJob]]:
        total = sum(len(s) for s in series)
        size = max(4, math.ceil(total / (2 * self.processes))) if self.processes > 1 else max(total, 1)
        out = []
        for s in series:
            s = sorted(s, key=lambda j: (j.time or "", j.path))
            out += [s[i:i + size] for i in range(0, len(s), size)]
        return [c for c in out if c]

    def solve(self, series: list[list[SolveJob]], catalog: CatalogStars, settings: SolveSettings,
              progress: Callable[[str], None] | None = None) -> dict[str, SolveResult]:
        chunks = self._chunks(series)
        out: dict[str, SolveResult] = {}
        if self.processes > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(min(self.processes, len(chunks))) as ex:
                for res in ex.map(_solve_chunk, chunks, [catalog] * len(chunks), [settings] * len(chunks)):
                    out.update(res)
        else:
            for c in chunks:
                out.update(_solve_chunk(c, catalog, settings))
        return out
