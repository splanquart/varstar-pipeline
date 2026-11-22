import datetime as dt

import numpy as np
import pytest

from varstar.fitsinfo import (BIAS, DARK, DARKFLAT, FLAT, LIGHT, UNKNOWN, folder_label_from_path,
                              normalize_imagetyp, parse_date_obs, read_frame)
from varstar.units import detect_adu_scale, detect_adu_scale_file

from conftest import write_frame


@pytest.mark.parametrize("raw,expected", [
    ("LIGHT", LIGHT), ("Light Frame", LIGHT), ("light", LIGHT), ("Dark Frame", DARK), ("DARK", DARK),
    ("FLAT", FLAT), ("Flat Field", FLAT), ("Bias Frame", BIAS), ("OFFSET", BIAS),
    ("DARKFLAT", DARKFLAT), ("Dark Flat", DARKFLAT), ("FLAT DARK", DARKFLAT), ("flat-dark", DARKFLAT),
    ("Master Dark", UNKNOWN), ("snapshot", UNKNOWN), (None, UNKNOWN),
])
def test_normalize_imagetyp(raw, expected):
    assert normalize_imagetyp(raw) == expected


def test_parse_date_obs_variants():
    assert parse_date_obs("2021-09-06T23:04:14.018") == dt.datetime(2021, 9, 6, 23, 4, 14, 18000)
    assert parse_date_obs("2021-09-06T23:04:14.0181234Z") == dt.datetime(2021, 9, 6, 23, 4, 14, 18123)
    assert parse_date_obs("garbage") is None
    assert parse_date_obs(None) is None


def test_folder_label():
    assert folder_label_from_path("light/H/SESSION_sept06/a.fits") == "sept06"
    assert folder_label_from_path("dark/300s/a.fits") is None


@pytest.mark.parametrize("utc,session", [
    ("2021-09-06T23:04:00", dt.date(2021, 9, 6)),
    ("2021-09-07T03:50:00", dt.date(2021, 9, 6)),
    ("2021-09-07T13:00:00", dt.date(2021, 9, 7)),
])
def test_session_date(tmp_path, utc, session):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, utc, filt="H")
    assert read_frame(p, tmp_path).session_date(-12) == session


def test_read_frame_header_fields(tmp_path):
    p = write_frame(tmp_path / "SESSION_a" / "x y.fits", "Light Frame", 300, "2021-09-06T23:04:00", filt="H",
                    temp=-11.7)
    f = read_frame(p, tmp_path)
    assert (f.imagetyp, f.exptime, f.gain, f.offset, f.xbin, f.filter) == (LIGHT, 300, 120, None, 2, "H")
    assert f.ccd_temp == pytest.approx(-11.7)
    assert f.folder_label == "a"
    assert f.issues == []


def test_missing_exptime_is_an_issue(tmp_path):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, "2021-09-06T23:04:00", filt="H")
    from astropy.io import fits
    with fits.open(p, mode="update") as h:
        del h[0].header["EXPTIME"]
        del h[0].header["EXPOSURE"]
    assert "EXPTIME/EXPOSURE absent" in read_frame(p, tmp_path).issues


def test_adu_scale_detection():
    rng = np.random.default_rng(0)
    assert detect_adu_scale((rng.integers(1, 16000, 10000) * 4).astype(np.uint16)) == 4
    assert detect_adu_scale(rng.integers(1, 65000, 10000).astype(np.uint16) | 1) == 1
    assert detect_adu_scale(np.zeros(10, dtype=np.uint16)) == 1
    assert detect_adu_scale(np.array([0.5, 4.0])) == 1


@pytest.mark.parametrize("scale", [1, 4, 16])
def test_adu_scale_from_file_with_bzero(tmp_path, scale):
    p = write_frame(tmp_path / "x.fits", "LIGHT", 300, "2021-09-06T23:04:00", filt="H", scale=scale,
                    shape=(40, 60))
    assert detect_adu_scale_file(p) == scale
