"""FITS file access for raw frames (synthetic files written by the tests)."""

import pytest
from astropy.io import fits

from fitsgen import write_frame
from varstar.adapters.fits_io import detect_adu_scale_file, read_frame
from varstar.domain.model import LIGHT


def test_read_frame_header_fields(tmp_path):
    p = write_frame(tmp_path / "SESSION_a" / "x y.fits", "Light Frame", 300, "2021-09-06T23:04:00", filt="H",
                    temp=-11.7)
    f = read_frame(p, tmp_path)
    assert (f.imagetyp, f.exptime, f.gain, f.offset, f.xbin, f.filter) == (LIGHT, 300, 120, None, 2, "H")
    assert f.ccd_temp == pytest.approx(-11.7)
    assert (f.path, f.rel_path, f.folder_label) == (p, "SESSION_a/x y.fits", "a")
    assert (f.naxis1, f.naxis2, f.bitpix) == (16, 12, 16)
    assert f.issues == []


def test_missing_exptime_is_an_issue(tmp_path):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, "2021-09-06T23:04:00", filt="H")
    with fits.open(p, mode="update") as h:
        del h[0].header["EXPTIME"]
        del h[0].header["EXPOSURE"]
    assert "EXPTIME/EXPOSURE absent" in read_frame(p, tmp_path).issues


def test_read_frame_does_not_touch_the_file(tmp_path):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, "2021-09-06T23:04:00", filt="H")
    before = (p.stat().st_mtime_ns, p.read_bytes())
    read_frame(p, tmp_path)
    assert (p.stat().st_mtime_ns, p.read_bytes()) == before


@pytest.mark.parametrize("scale", [1, 4, 16])
def test_adu_scale_from_file_with_bzero(tmp_path, scale):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, "2021-09-06T23:04:00", filt="H", scale=scale,
                    shape=(40, 60))
    assert detect_adu_scale_file(p) == scale
