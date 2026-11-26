"""Image files: Siril-normalised floats, float32 and mask writers, minimal XISF reader."""

import numpy as np
import pytest
from astropy.io import fits

from fitsgen import write_xisf
from varstar.adapters.images import (NORM_SCALE, XisfError, read_adu, read_siril_output, read_xisf, write_float32,
                                     write_mask)


def test_read_xisf(tmp_path):
    data = np.random.default_rng(3).random((30, 50)).astype(np.float32)
    np.testing.assert_array_equal(read_xisf(write_xisf(tmp_path / "m.xisf", data)), data)
    with pytest.raises(XisfError, match="compressée"):
        read_xisf(write_xisf(tmp_path / "c.xisf", data, compression="zlib"))
    (tmp_path / "bad.xisf").write_bytes(b"NOTXISF!")
    with pytest.raises(XisfError, match="signature"):
        read_xisf(tmp_path / "bad.xisf")


def test_read_siril_output_rejects_unnormalised(tmp_path):
    fits.PrimaryHDU(np.full((4, 4), 0.01, np.float32)).writeto(tmp_path / "ok.fits")
    fits.PrimaryHDU(np.full((4, 4), 500, np.float32)).writeto(tmp_path / "bad.fits")
    data, _ = read_siril_output(tmp_path / "ok.fits")
    assert np.allclose(data, 0.01 * NORM_SCALE)
    with pytest.raises(ValueError):
        read_siril_output(tmp_path / "bad.fits")


def test_read_adu_uses_vsnorm(tmp_path):
    hdu = fits.PrimaryHDU(np.full((4, 4), 0.5, np.float32))
    hdu.header["VSNORM"] = NORM_SCALE
    hdu.writeto(tmp_path / "m.fits")
    data, header = read_adu(tmp_path / "m.fits")
    assert data.dtype == np.float32 and np.allclose(data, 0.5 * NORM_SCALE) and header["VSNORM"] == NORM_SCALE


def test_writers_float32_and_uint8(tmp_path):
    header = fits.Header()
    header["BZERO"] = 32768
    header["VSLEVEL"] = "A"
    write_float32(tmp_path / "out/a.fits", np.arange(6, dtype=np.float64).reshape(2, 3), header)
    with fits.open(tmp_path / "out/a.fits") as hdul:
        assert hdul[0].header["BITPIX"] == -32 and "BZERO" not in hdul[0].header
        assert hdul[0].header["VSLEVEL"] == "A"
    write_mask(tmp_path / "m/b.fits", np.array([[0, 3]]))
    assert fits.getdata(tmp_path / "m/b.fits").dtype == np.uint8
    assert not list(tmp_path.rglob("*.tmp"))
