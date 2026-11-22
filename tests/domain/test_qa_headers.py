"""Header checks (H*) on inventories built in memory."""

import datetime as dt

import pytest

from builders import inventory_of, make_frame, make_settings, sept06_frames, series_frames, two_nights_frames
from varstar.domain.model import DARK, FLAT, LIGHT
from varstar.domain.qa.headers import header_checks
from varstar.domain.qa.model import SEVERITIES


def checks(frames, cfg=None, **kw):
    inv = inventory_of(frames, cfg, **kw)
    return header_checks(inv, inv.config)


def by_check(items, check):
    return [i for i in items if i.check == check]


def test_items_are_sorted_by_severity_then_check():
    items = checks(two_nights_frames(light_temp=-9.5))
    keys = [(SEVERITIES.index(i.severity), i.check) for i in items]
    assert keys == sorted(keys)


def test_h1_h2_mirror_the_inventory_anomalies():
    frames = sept06_frames() + [make_frame(DARK, dt.datetime(2021, 9, 7, 3, 0), exptime=180,
                                           rel_path="dark/darklight/300s/x.fits")]
    items = checks(frames)
    (h1,) = by_check(items, "H1")
    assert h1.files == ["dark/darklight/300s/x.fits"] and h1.data["details"]
    assert by_check(items, "H2") and "OFFSET" in by_check(items, "H2")[0].message


def test_h3_temperature_whole_night_and_first_frames():
    (h3,) = by_check(checks(two_nights_frames(light_temp=-9.5)), "H3")
    assert h3.severity == "warning" and h3.data == {"n_bad": 3, "n": 3, "max_delta_c": 2.5}
    assert "toute la nuit" in h3.message and "sept06" in h3.message

    warm_start = [f for f in two_nights_frames() if not f.rel_path.startswith("light/H/SESSION_sept06")]
    lights = series_frames("light/H/SESSION_sept06", LIGHT, 300, "2021-09-06T23:04:00", 4, step_s=600, filter="H")
    lights[0].ccd_temp = -8.0
    (h3,) = by_check(checks(warm_start + lights), "H3")
    assert "dont les 1 première(s)" in h3.message and h3.files == [lights[0].rel_path]


def test_h3_flats_are_only_info():
    frames = sept06_frames()
    for f in frames:
        if f.imagetyp == FLAT:
            f.ccd_temp = -5.0
    (h3,) = by_check(checks(frames), "H3")
    assert h3.severity == "info" and h3.message.startswith("flats")


def test_h4_software_versions():
    frames = two_nights_frames()
    for f in frames:
        f.swcreate = "N.I.N.A. 2.0" if f.rel_path.startswith(("light/H/SESSION_sept02", "dark/a")) else "N.I.N.A. 3.0"
    frames[0].swcreate = "N.I.N.A. 2.0"
    msgs = [i.message for i in by_check(checks(frames, make_settings(matching={"dark_same_night": False})), "H4")]
    assert any("change selon les nuits" in m for m in msgs)
    assert any("autre version" in m for m in msgs)


def test_h5_focus_of_flats():
    frames = sept06_frames()
    for f in frames:
        f.focpos = 5000 if f.imagetyp == LIGHT else (5600 if f.imagetyp == FLAT else None)
    items = by_check(checks(frames), "H5")
    assert len(items) == 2 and {i.severity for i in items} == {"info"}
    assert items[0].data["flat_focpos"] == 5600
    for f in frames:
        if f.imagetyp == FLAT:
            f.focpos = 5050
    assert by_check(checks(frames), "H5") == []


def test_h6_pointing():
    frames = sept06_frames()
    for f in frames:
        if f.imagetyp == LIGHT:
            f.ra, f.dec, f.object = 60.0, 36.0, "NGC 1499"
    far = next(f for f in frames if f.imagetyp == LIGHT and f.filter == "S")
    far.ra = 61.5
    (h6,) = by_check(checks(frames), "H6")
    assert h6.files == [far.rel_path] and "S" in h6.message and h6.data["max_arcmin"] > 60
    assert by_check(checks(frames, adu_scale=None), "H6")   # the pixel scale comes from the headers


def test_h7_weather():
    frames = two_nights_frames() + series_frames(
        "light/H/SESSION_sept12", LIGHT, 300, "2021-09-12T23:00:00", 2, step_s=600, filter="H",
        ambtemp=10.0, dewpoint=9.0, cloudcvr=80.0)
    sev = {i.severity: i for i in by_check(checks(frames), "H7")}
    assert set(sev) == {"warning", "info"}
    assert "rosée" in sev["warning"].message and "nuageuse" in sev["info"].message


def test_h8_darks_pointing_at_the_sky():
    frames = sept06_frames()
    for f in frames:
        if f.imagetyp == DARK:
            f.centalt = 45.0
    (h8,) = by_check(checks(frames), "H8")
    assert "45–45°" in h8.message


@pytest.mark.parametrize("n_darks, expected", [(5, True), (15, False)])
def test_h9_number_of_darks(n_darks, expected):
    frames = [f for f in sept06_frames() if not f.rel_path.startswith("dark/darklight")]
    frames += series_frames("dark/darklight/300s", DARK, 300, "2021-09-07T02:00:00", n_darks, step_s=310)
    darks = [i for i in by_check(checks(frames), "H9") if "darks seulement" in i.message]
    assert bool(darks) is expected
