"""``SirilCalibrator`` parts that need no Siril: command, provenance header, thermal dark, level C copy."""

import numpy as np
from astropy.io import fits

from fitsgen import write_frame
from varstar.adapters.images import NORM_SCALE
from varstar.adapters.siril_calibrator import SirilCalibrator, calibrate_command, provenance_header, thermal_dark
from varstar.domain.calibration import CalibrationSensor
from varstar.domain.plan import DarkflatRef, DarkRef, FlatRef, GroupPlan


def group(files=(), dark=None, flat=None, level="A"):
    return GroupPlan("2021-09-06_H_300s", "2021-09-06", "s1", "H", 300.0, level, [], list(files), dark, flat,
                     DarkflatRef("df", "/lib/df.fits", 1.13) if flat else None,
                     "/w/masks/bpm_x.fits" if dark else None, None)


def test_calibrate_command():
    direct = DarkRef("d", "/lib/d.fits", "direct", 8, 300.0)
    scaled = DarkRef("d", "/lib/d.fits", "scaled", 8, 180.0, 512.345, "o")
    flat = FlatRef("f", "/lib/f.fits", 5)
    assert calibrate_command(group(dark=direct, flat=flat)) == "calibrate l -dark=mdark -flat=mflat -prefix=pp_"
    assert calibrate_command(group(dark=scaled)) == 'calibrate l -bias="=512.35" -dark=mdark -opt=exp -prefix=pp_'
    assert calibrate_command(group(flat=flat)) == "calibrate l -flat=mflat -prefix=pp_"


def test_provenance_header(tmp_path):
    raw = write_frame(tmp_path / "raw/a b.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    siril = fits.Header()
    siril.add_history("Calibration with dark")
    g = group([raw], DarkRef("d1", "/lib/d.fits", "direct", 8, 300.0), FlatRef("f1", "/lib/f.fits", 5))
    h = provenance_header(raw, g, siril, CalibrationSensor(4, 0.25))
    assert "BZERO" not in h and h["BUNIT"] == "ADU" and h["VSRAW"] == str(raw) and h["VSLEVEL"] == "A"
    assert (h["VSDARK"], h["VSDKMODE"], h["VSFLAT"], h["VSDFLAT"]) == ("d1", "direct", "f1", "df")
    assert (h["VSBPM"], h["VSAMPG"], h["VSADUSC"], h["EGAINEFF"]) == ("bpm_x.fits", "NONE", 4, 0.25)
    assert h["FILTER"] == "H" and any("siril: Calibration" in c for c in h["HISTORY"])
    bare = provenance_header(raw, group([raw], level="C"), None, CalibrationSensor())
    assert bare["VSDARK"] == "NONE" and "VSADUSC" not in bare and "EGAINEFF" not in bare


def test_thermal_dark_removes_the_offset(tmp_path):
    hdu = fits.PrimaryHDU(np.full((4, 4), 600 / NORM_SCALE, np.float32))
    hdu.header["VSNORM"] = NORM_SCALE
    hdu.writeto(tmp_path / "dark.fits")
    thermal_dark(tmp_path / "dark.fits", 500.0, tmp_path / "thermal.fits")
    with fits.open(tmp_path / "thermal.fits") as hdul:
        assert np.allclose(hdul[0].data * NORM_SCALE, 100, atol=1e-2) and "VSNORM" not in hdul[0].header


def test_level_c_without_master_copies_raw_values_as_float32(tmp_path):
    data = (np.arange(12 * 16).reshape(12, 16) * 4).astype(np.uint16)
    raw = write_frame(tmp_path / "raw/a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H", data=data)
    out = tmp_path / "w/calibrated/a.fits"
    SirilCalibrator(siril=None, workdir=tmp_path / "w").calibrate(group([raw], level="C"), {str(raw): out},
                                                                  CalibrationSensor(4, 0.25))
    with fits.open(out) as hdul:
        assert hdul[0].header["BITPIX"] == -32 and hdul[0].header["VSLEVEL"] == "C"
        assert np.array_equal(hdul[0].data, data.astype(np.float32))
