"""Derived properties of the domain model (in memory, no file)."""

import datetime as dt

import pytest

from builders import (SENSOR, T0, make_frame, make_frames, make_frameset, make_inventory, make_light_group,
                      make_sensor_info)
from varstar.domain.model import DARK, FLAT, LIGHT, Match, SensorKey


def test_sensor_key_label():
    assert SENSOR.label() == "ASI Camera (1) (ASCO | gain 120 | offset ? | bin 2x2 | 4144x2822"
    assert SensorKey("cam", None, 30.0, 1, 1, 10, 20).label() == "cam | gain ? | offset 30 | bin 1x1 | 10x20"


def test_frame_sensor_key_and_session_date():
    f = make_frame(LIGHT, dt.datetime(2021, 9, 7, 3, 0))
    assert f.sensor_key == SENSOR
    assert f.session_date() == dt.date(2021, 9, 6)            # 03:00 belongs to the night of the 6th
    assert f.session_date(offset_hours=0) == dt.date(2021, 9, 7)
    assert make_frame(LIGHT, None).session_date() is None


def test_frameset_time_span_and_temperature():
    s = make_frameset(DARK, n=3, start=T0, step_s=600, ccd_temp=-12.0)
    assert (s.n, s.start, s.end) == (3, T0, T0 + dt.timedelta(minutes=20))
    assert s.mid == T0 + dt.timedelta(minutes=10)
    assert s.mean_temp == -12.0
    s.frames[0].ccd_temp = None
    assert s.mean_temp == -12.0                                  # missing temperatures ignored
    for f in s.frames:
        f.ccd_temp = None
    assert s.mean_temp is None


def test_frameset_folders_and_labels():
    frames = [make_frame(FLAT, T0, rel_path="flat/SESSION_s1/a.fits", folder_label="s1"),
              make_frame(FLAT, T0, rel_path="dark/oops/b.fits")]
    s = make_frameset(FLAT, n=1, filter="H")
    s.frames = frames
    assert s.folders == ["dark/oops", "flat/SESSION_s1"]
    assert s.folder_labels == ["s1"]


@pytest.mark.parametrize("dark,flat,darkflat,level", [
    ("A", "A", "A", "A"),
    ("A", "A", None, "A"),        # no dark-flat match: not applicable
    ("B", "A", "A", "B"),
    ("A", "C", None, "C"),
    (None, "A", "A", "C"),        # missing dark match → C
])
def test_light_group_level_is_the_worst_component(dark, flat, darkflat, level):
    m = lambda lv, why: Match(lv, [], [why]) if lv else None  # noqa: E731
    g = make_light_group(dark=m(dark, "d"), flat=m(flat, "f"), darkflat=m(darkflat, "df"))
    assert g.level == level
    assert g.reasons == [r for lv, r in ((dark, "d"), (flat, "f"), (darkflat, "df")) if lv]


def test_light_group_temperatures():
    g = make_light_group(n=2, ccd_temp=-11.0, set_temp=-12.0)
    assert (g.mean_temp, g.mean_set_temp, g.n) == (-11.0, -12.0, 2)
    assert g.start < g.end


def test_sensor_info_derived_values():
    s = make_sensor_info()
    assert s.effective_gain == pytest.approx(0.25056, abs=1e-5)
    assert s.saturation_adu == 65532
    assert s.pixel_scale_arcsec == pytest.approx(3.485, abs=1e-3)
    assert make_sensor_info(adu_scale=None).effective_gain is None
    assert make_sensor_info(adu_scale=None).saturation_adu is None
    assert make_sensor_info(focallen=None).pixel_scale_arcsec is None


def test_inventory_levels_labels_and_counts():
    a = Match("A")
    g1 = make_light_group(filter="H", label="s1", dark=a, flat=a)
    g2 = make_light_group(filter="H", label="s1", exptime=120, dark=Match("B", [], ["x"]), flat=a)
    g3 = make_light_group(start=T0 + dt.timedelta(days=3), filter="S", label=None, dark=a, flat=a)
    inv = make_inventory([g1, g2, g3], frames=[*g1.frames, *make_frames(2, DARK)])
    assert inv.levels_by_session_filter() == {(dt.date(2021, 9, 6), "H"): "B", (dt.date(2021, 9, 9), "S"): "A"}
    assert inv.session_label(dt.date(2021, 9, 6)) == "2021-09-06 (s1)"
    assert inv.session_label(dt.date(2021, 9, 9)) == "2021-09-09"
    assert inv.counts_by_type() == {LIGHT: 3, DARK: 2}
