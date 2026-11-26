"""N.I.N.A. keyword interpretation on plain dicts (no FITS file)."""

import datetime as dt
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from varstar.domain.headers import folder_label_from_path, frame_from_header, normalize_imagetyp, parse_date_obs
from varstar.domain.model import BIAS, DARK, DARKFLAT, FLAT, LIGHT, UNKNOWN

ROOT = Path("/raw")

NINA_LIGHT = {
    "SIMPLE": True, "BITPIX": 16, "NAXIS": 2, "NAXIS1": 4144, "NAXIS2": 2822, "IMAGETYP": "LIGHT",
    "EXPOSURE": 300.0, "EXPTIME": 300.0, "DATE-OBS": "2021-09-06T23:04:14.018", "XBINNING": 2,
    "YBINNING": 2, "GAIN": 120, "EGAIN": 1.00224268436432, "XPIXSZ": 9.26, "INSTRUME": "ASI Camera (1) (ASCO",
    "SET-TEMP": -12.0, "CCD-TEMP": -11.7, "FOCALLEN": 274.0, "FILTER": "H", "OBJECT": "NGC 1499",
    "RA": 60.1, "DEC": 36.4, "FOCPOS": 5210, "FOCTEMP": 8.5, "AMBTEMP": 9.1, "DEWPOINT": 4.2,
    "SWCREATE": "N.I.N.A. 1.10",
}


def frame(header, rel="light/H/SESSION_sept06/2021-09-06_23-04-14_NGC 1499_H_300.00s_0000.fits"):
    return frame_from_header(ROOT / rel, header, ROOT)


# --------------------------------------------------------------------------- examples


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
    assert parse_date_obs("2021-09-07T01:04:14+02:00") == dt.datetime(2021, 9, 6, 23, 4, 14)
    assert parse_date_obs("garbage") is None
    assert parse_date_obs("") is None
    assert parse_date_obs(None) is None


def test_folder_label():
    assert folder_label_from_path("light/H/SESSION_sept06/a.fits") == "sept06"
    assert folder_label_from_path("SESSION_a/session-b/x.fits") == "b"       # innermost wins
    assert folder_label_from_path("dark/300s/a.fits") is None


def test_nina_light_header():
    f = frame(NINA_LIGHT)
    assert (f.imagetyp, f.exptime, f.gain, f.offset, f.xbin, f.ybin, f.filter) == (LIGHT, 300, 120, None, 2, 2, "H")
    assert (f.naxis1, f.naxis2, f.bitpix, f.instrument) == (4144, 2822, 16, "ASI Camera (1) (ASCO")
    assert f.ccd_temp == pytest.approx(-11.7) and f.set_temp == -12.0
    assert f.date_obs == dt.datetime(2021, 9, 6, 23, 4, 14, 18000)
    assert (f.focpos, f.foctemp, f.ambtemp, f.dewpoint, f.swcreate) == (5210, 8.5, 9.1, 4.2, "N.I.N.A. 1.10")
    assert f.folder_label == "sept06"
    assert f.rel_path.startswith("light/H/SESSION_sept06/")
    assert f.issues == []


@pytest.mark.parametrize("utc,session", [
    ("2021-09-06T23:04:00", dt.date(2021, 9, 6)),
    ("2021-09-07T03:50:00", dt.date(2021, 9, 6)),
    ("2021-09-07T13:00:00", dt.date(2021, 9, 7)),
])
def test_session_date(utc, session):
    assert frame({**NINA_LIGHT, "DATE-OBS": utc}).session_date(-12) == session


def test_keyword_aliases_and_defaults():
    h = {k: v for k, v in NINA_LIGHT.items() if k not in ("EXPTIME", "XBINNING", "YBINNING", "CCD-TEMP", "FOCPOS",
                                                          "INSTRUME", "IMAGETYP")}
    h.update({"FRAMETYP": "Dark Frame", "CCD_TEMP": -12.3, "FOCUSPOS": 5000})
    f = frame(h)
    assert f.imagetyp == DARK and f.imagetyp_raw == "Dark Frame"
    assert f.exptime == 300.0                       # EXPOSURE used when EXPTIME is absent
    assert (f.xbin, f.ybin) == (1, 1)               # binning defaults to 1
    assert f.ccd_temp == -12.3 and f.focpos == 5000
    assert f.instrument == "?"


def test_empty_or_unreadable_values_are_missing():
    f = frame({**NINA_LIGHT, "GAIN": "", "OFFSET": "n/a", "FILTER": "  H  ", "OBJECT": ""})
    assert f.gain is None and f.offset is None
    assert f.filter == "H" and f.object is None


@pytest.mark.parametrize("change,issue", [
    ({"IMAGETYP": "snapshot"}, "IMAGETYP non reconnu : 'snapshot'"),
    ({"EXPTIME": None, "EXPOSURE": None}, "EXPTIME/EXPOSURE absent"),
    ({"DATE-OBS": "hier soir"}, "DATE-OBS absent ou illisible"),
    ({"NAXIS1": 0}, "dimensions absentes (NAXIS1/NAXIS2)"),
])
def test_blocking_header_problems_are_listed(change, issue):
    h = {k: v for k, v in {**NINA_LIGHT, **change}.items() if v is not None}
    assert frame(h).issues == [issue]


def test_path_outside_root_is_kept_absolute():
    f = frame_from_header(Path("/elsewhere/x.fits"), NINA_LIGHT, ROOT)
    assert f.rel_path == "/elsewhere/x.fits" and f.folder_label is None


# --------------------------------------------------------------------------- properties

KEYWORDS = {"LIGHT": LIGHT, "DARK": DARK, "FLAT": FLAT, "BIAS": BIAS, "DARKFLAT": DARKFLAT, "FLATDARK": DARKFLAT}


@st.composite
def imagetyp_spelling(draw):
    """A canonical IMAGETYP with random case, separators anywhere and an optional ' Frame' suffix."""
    word = draw(st.sampled_from(sorted(KEYWORDS)))
    seps = draw(st.lists(st.sampled_from(["", "", " ", "_", "-"]), min_size=len(word), max_size=len(word)))
    text = "".join(c + sep for c, sep in zip(word, seps[:-1] + [""]))
    text = "".join(c.lower() if draw(st.booleans()) else c for c in text)
    return word, text + draw(st.sampled_from(["", " Frame", " frames", " FRAME"]))


@given(imagetyp_spelling())
def test_imagetyp_ignores_case_separators_and_frame_suffix(spelled):
    word, text = spelled
    assert normalize_imagetyp(text) == KEYWORDS[word]
    assert normalize_imagetyp("Master " + text) == UNKNOWN


@given(st.datetimes(min_value=dt.datetime(1990, 1, 1), max_value=dt.datetime(2100, 1, 1)), st.booleans())
def test_date_obs_round_trip(d, zulu):
    assert parse_date_obs(d.isoformat() + ("Z" if zulu else "")) == d


@given(st.datetimes(min_value=dt.datetime(1990, 1, 2), max_value=dt.datetime(2100, 1, 1)),
       st.integers(-12, 14))
def test_date_obs_with_offset_is_converted_to_utc(d, hours):
    tz = dt.timezone(dt.timedelta(hours=hours))
    local = (d + dt.timedelta(hours=hours)).replace(tzinfo=tz)
    assert parse_date_obs(local.isoformat()) == d
