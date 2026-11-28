"""Internal plate solver on FITS files (synthetic sky), sequential and in worker processes."""

import numpy as np
import pytest
from astropy.io import fits

from builders import make_solution, synthetic_sky
from varstar.adapters.plate_solver import InternalPlateSolver
from varstar.domain.astrometry import SolveHint, SolveJob, SolveSettings

SHAPE = (300, 450)
SCALE = 3.4877


@pytest.fixture(scope="module")
def field(tmp_path_factory):
    d = tmp_path_factory.mktemp("solver")
    truth = make_solution(60.44, 36.11, scale_arcsec=SCALE, width=SHAPE[1], height=SHAPE[0], orientation_deg=180.6,
                          parity=1)
    img, cat = synthetic_sky(truth, SHAPE, n=700)
    paths = []
    for i in range(5):
        p = d / f"img_{i}.fits"
        fits.writeto(p, (img + np.float32(i)).astype(np.float32))
        paths.append(str(p))
    hint = SolveHint(60.45, 36.10, SCALE, SHAPE[1], SHAPE[0])
    jobs = [SolveJob(p, hint, f"2021-09-06T23:{i:02d}:00") for i, p in enumerate(paths)]
    return truth, cat, jobs, d


@pytest.mark.parametrize("processes", [1, 3])
def test_solves_series(field, processes):
    truth, cat, jobs, d = field
    missing = SolveJob(str(d / "absent.fits"), jobs[0].hint, "2021-09-06T23:59:00")
    solver = InternalPlateSolver(processes)
    out = solver.solve([jobs[:3], jobs[3:] + [missing]], cat, SolveSettings(sip_order=0, min_matches=15))
    assert set(out) == {j.path for j in jobs} | {missing.path}
    for j in jobs:
        r = out[j.path]
        assert r.ok and r.solution.parity == 1
        x, y = r.solution.world_to_pixel(*truth.crval)
        assert abs(x - (SHAPE[1] - 1) / 2) < 0.3 and abs(y - (SHAPE[0] - 1) / 2) < 0.3
    assert not out[missing.path].ok and "illisible" in out[missing.path].reason
    assert solver.name().startswith("interne")


def test_chunks_keep_series_apart():
    hint = SolveHint(0, 0, 1, 10, 10)
    a = [SolveJob(f"a{i}", hint, f"t{i}") for i in range(10)]
    b = [SolveJob(f"b{i}", hint, f"t{i}") for i in range(3)]
    chunks = InternalPlateSolver(2)._chunks([a, b])
    assert all(len({j.path[0] for j in c}) == 1 for c in chunks)
    assert sum(len(c) for c in chunks) == 13 and len(InternalPlateSolver(1)._chunks([a, b])) == 2
