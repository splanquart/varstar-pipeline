"""``DiskMasterRepository``: the library of masters on disk."""

import numpy as np
from astropy.io import fits

from builders import make_frames
from varstar.adapters.images import NORM_SCALE
from varstar.adapters.library import DiskMasterRepository
from varstar.domain.model import DARK

META = {"id": "abc123", "kind": "dark", "label": "DARK_g120", "rejection": {"name": "sigma clipping"},
        "siril_version": "1.4.5", "created": "2026-11-26T09:00:00", "n_frames": 3, "median_adu": 500.0}


def test_publish_find_and_load(tmp_path):
    repo = DiskMasterRepository(tmp_path / "lib")
    assert repo.find("dark", "abc123") is None
    d = repo.new_dir("dark", "DARK_g120", "abc123")
    assert d == tmp_path / "lib" / "darks" / "DARK_g120_abc123"
    frames = make_frames(3, DARK, filter=None)
    repo.publish(d, np.full((4, 6), 0.5, np.float32), fits.Header(), META, frames)
    rec = repo.find("dark", "abc123")
    assert rec.meta == META and rec.dir == d and repo.find("flat", "abc123") is None
    h = fits.getheader(rec.path)
    assert (h["VSKIND"], h["VSID"], h["VSNFRAM"], h["VSNORM"]) == ("DARK", "abc123", 3, NORM_SCALE)
    assert len(h["HISTORY"]) == 3
    assert np.allclose(repo.load_adu(rec), 0.5 * NORM_SCALE)
    assert repo.load_unstable(rec) is None
    fits.PrimaryHDU(np.ones((4, 6), np.uint8)).writeto(rec.unstable_path)
    assert repo.load_unstable(rec).sum() == 24


def test_incomplete_master_is_ignored(tmp_path):
    repo = DiskMasterRepository(tmp_path / "lib")
    d = repo.new_dir("flat", "FLAT_H", "f1")
    (d / "master.json").write_text("{}")             # master.fits missing (interrupted build)
    assert repo.find("flat", "f1") is None
