"""Inventory rules on frames built in memory (sessions, series, matching, A/B/C levels, anomalies)."""

import datetime as dt

from hypothesis import given
from hypothesis import strategies as st

from builders import SENSOR, make_frame, make_scan, make_settings, sept06_frames, series_frames
from varstar.domain.inventory import adu_scale_samples, build_inventory, build_sets, darks_todo, usable_frames
from varstar.domain.model import DARK, DARKFLAT, FLAT, LIGHT, UNKNOWN, AduScaleReading, SensorKey

SCALE_4 = {SENSOR: AduScaleReading("light/H/SESSION_sept06/x.fits", scale=4)}


def inventory(frames, cfg=None, readings=SCALE_4, **scan):
    return build_inventory(make_scan(frames, **scan), cfg or make_settings(), readings, dt.datetime(2026, 1, 1))


def levels(inv):
    return {(s.isoformat(), f): lv for (s, f), lv in inv.levels_by_session_filter().items()}


def group(inv, filt, session=None):
    gs = [g for g in inv.light_groups if g.filter == filt and (session is None or g.session.isoformat() == session)]
    assert len(gs) == 1
    return gs[0]


def without(frames, folder="", exptime=None, filter=None):
    """Drop the frames of a folder (optionally only one exposure or filter)."""
    return [f for f in frames if not (f.rel_path.startswith(folder) and (exptime is None or f.exptime == exptime)
                                      and (filter is None or f.filter == filter))]


# --------------------------------------------------------------------------- levels and matching


def test_complete_session_is_level_a():
    inv = inventory(sept06_frames())
    assert levels(inv) == {("2021-09-06", "H"): "A", ("2021-09-06", "S"): "A"}
    g = group(inv, "H")
    assert g.n == 3 and g.dark.n_frames == 4 and g.flat.sets[0].exptime == 1.13
    assert g.darkflat.sets[0].exptime == 1.13
    assert darks_todo(inv) == []


def test_header_wins_over_folder():
    # A 180 s dark misfiled in a 300s folder, and an H light stored in an S folder.
    frames = sept06_frames() + [
        make_frame(DARK, dt.datetime(2021, 9, 7, 3, 0), exptime=180, rel_path="dark/darklight/300s/x.fits"),
        make_frame(LIGHT, dt.datetime(2021, 9, 7, 1, 0), filter="H", rel_path="light/S/SESSION_sept06/y.fits",
                   folder_label="sept06"),
    ]
    inv = inventory(frames)
    codes = {a.code for a in inv.anomalies}
    assert {"folder_exptime", "folder_filter"} <= codes
    assert group(inv, "H").n == 4
    assert any(s.exptime == 180 for s in inv.calibration_sets[DARK])


def test_missing_exact_darkflat_gives_b():
    frames = without(sept06_frames(), "flat/SESSION_sept06", filter="H") + [
        *series_frames("flat/SESSION_sept06", FLAT, 1.61, "2021-09-07T13:21:00", 3, step_s=2, filter="H"),
        *series_frames("dark/darkflat", DARKFLAT, 1.69, "2021-09-07T13:40:00", 3, step_s=2),
    ]
    inv = inventory(frames)
    g = group(inv, "H")
    assert g.level == "B" and g.darkflat.level == "B" and g.darkflat.sets[0].exptime == 1.69
    assert [(t.kind, t.exptime) for t in darks_todo(inv)] == [("darkflat", 1.61)]


def test_darkflat_tolerance():
    frames = without(sept06_frames(), "dark/darkflat", exptime=1.13) + series_frames(
        "dark/darkflat", DARKFLAT, 1.16, "2021-09-08T18:25:00", 3, step_s=2)
    assert group(inventory(frames), "H").darkflat.level == "A"


def test_no_darkflat_at_all_makes_flats_unusable():
    g = group(inventory(without(sept06_frames(), "dark/")), "H")
    assert g.flat.level == "C" and g.level == "C"
    assert "flats écartés faute de dark de flat" in g.flat.reasons


def test_darks_from_another_exposure_are_scaled_b():
    frames = without(sept06_frames(), "dark/darklight") + [
        *series_frames("dark/darklight/180s", DARK, 180, "2021-09-07T02:00:00", 3, step_s=200),
        *series_frames("dark/darklight/600s", DARK, 600, "2021-09-07T03:00:00", 3, step_s=610),
    ]
    inv = inventory(frames)
    g = group(inv, "H")
    assert g.dark.level == "B" and g.dark.sets[0].exptime == 180  # shorter darks preferred
    assert "mise à l'échelle" in g.dark.reasons[0]
    assert [(t.kind, t.exptime) for t in darks_todo(inv)] == [("dark", 300.0)]
    no_scaling = make_settings(matching={"allow_dark_scaling": False})
    assert group(inventory(frames, no_scaling), "H").dark.level == "C"


def test_dark_temperature_mismatch_is_b():
    frames = without(sept06_frames(), "dark/darklight") + series_frames(
        "dark/darklight/300s", DARK, 300, "2021-09-07T02:00:00", 3, step_s=310, ccd_temp=-5.0)
    g = group(inventory(frames), "H")
    assert g.dark.level == "B" and "ΔT" in g.dark.reasons[0]


def test_darks_of_another_night_are_borrowed_at_level_b():
    frames = without(sept06_frames(), "dark/darklight") + series_frames(
        "dark/darklight/300s", DARK, 300, "2021-09-10T02:00:00", 4, step_s=310)
    inv = inventory(frames)
    g = group(inv, "H")
    assert g.dark.level == "B" and g.dark.borrowed_night and "même nuit" in g.dark.reasons[0]
    assert darks_todo(inv) == []          # darks of a past night cannot be redone


def test_sensor_key_must_match():
    other = SensorKey(SENSOR.instrument, 200.0, None, 2, 2, 4144, 2822)
    frames = without(sept06_frames(), "dark/darklight") + series_frames(
        "dark/darklight/300s", DARK, 300, "2021-09-07T02:00:00", 3, step_s=310, sensor=other)
    inv = inventory(frames)
    assert group(inv, "H").dark.level == "C"
    assert any(a.code == "several_sensors" for a in inv.anomalies)


def test_session_without_flats_is_c_then_b_with_same_setup():
    frames = sept06_frames() + series_frames("light/H/SESSION_sept02", LIGHT, 300, "2021-09-02T23:28:00", 3,
                                             step_s=600, filter="H")
    g = group(inventory(frames), "H", "2021-09-02")
    assert g.flat.level == "C" and g.level == "C"
    assert "same_optical_setup" in g.flat.reasons[0]

    same_setup = {"same_optical_setup": [["2021-09-02", "2021-09-07"]]}
    g = group(inventory(frames, make_settings(matching=same_setup)), "H", "2021-09-02")
    assert g.flat.level == "B" and g.level == "B"
    assert g.flat.sets[0].start.date() == dt.date(2021, 9, 7)

    too_old = make_settings(matching={**same_setup, "max_flat_age_days": 3})
    assert group(inventory(frames, too_old), "H", "2021-09-02").flat.level == "C"


def test_flats_in_same_session_folder_but_days_before():
    """Question Q3: flats stored in SESSION_aout31 but taken on 29/08."""
    frames = [
        *series_frames("light/H/SESSION_aout31", LIGHT, 300, "2021-08-31T23:28:00", 2, step_s=600, filter="H"),
        *series_frames("flat/SESSION_aout31", FLAT, 1.13, "2021-08-29T16:09:00", 3, step_s=2, filter="H"),
        *series_frames("dark/300s", DARK, 300, "2021-09-01T02:18:00", 3, step_s=310),
        *series_frames("dark/darkflat", DARKFLAT, 1.13, "2021-09-08T18:25:00", 3, step_s=2),
    ]
    g = group(inventory(frames), "H")
    assert g.level == "A" and "Q3" in g.flat.reasons[0] and "avant" in g.flat.reasons[0]
    untrusted = make_settings(matching={"trust_session_folder_flats": False})
    assert group(inventory(frames, untrusted), "H").flat.level == "C"


def test_flats_of_other_filter_not_used():
    inv = inventory(without(sept06_frames(), "flat/SESSION_sept06", filter="S"))
    assert group(inv, "S").flat.level == "C"
    assert group(inv, "H").flat.level == "A"


# --------------------------------------------------------------------------- anomalies and sensors


def test_offset_missing_warning_once():
    inv = inventory(sept06_frames())
    assert sum(a.code == "offset_missing" for a in inv.anomalies) == 1


def test_bad_header_frame_excluded():
    bad = make_frame(UNKNOWN, dt.datetime(2021, 9, 6, 23, 30), rel_path="light/H/z.fits")
    bad.issues.append("IMAGETYP non reconnu : 'WEIRD'")
    inv = inventory(sept06_frames() + [bad])
    assert group(inv, "H").n == 3
    (a,) = [a for a in inv.anomalies if a.code == "bad_header"]
    assert a.files == ["light/H/z.fits"] and "exclu" in a.message


def test_scan_facts_become_anomalies():
    inv = inventory(sept06_frames(), unreadable=["light/broken.fits (OSError)"], empty_dirs=["bias/SESSION_1"],
                    ignored_dirs=["removed"], other_extensions={".xisf": 2})
    by_code = {a.code: a for a in inv.anomalies}
    assert by_code["unreadable"].severity == "error" and by_code["unreadable"].files == ["light/broken.fits (OSError)"]
    assert by_code["empty_dirs"].files == ["bias/SESSION_1"] and by_code["ignored_dirs"].files == ["removed"]
    assert inv.other_extensions == {".xisf": 2}


def test_lights_without_filter_get_question_mark():
    frames = sept06_frames() + [make_frame(LIGHT, dt.datetime(2021, 9, 6, 23, 50), filter=None,
                                           rel_path="light/x.fits")]
    inv = inventory(frames)
    assert any(a.code == "no_filter" for a in inv.anomalies)
    assert group(inv, "?").n == 1


def test_sensor_info_from_adu_reading():
    (s,) = inventory(sept06_frames()).sensors
    assert (s.adu_scale, s.adu_scale_source) == (4, "light/H/SESSION_sept06/x.fits")
    assert abs(s.effective_gain - 1.00224268436432 / 4) < 1e-9
    assert s.saturation_adu == 65532


def test_sensor_info_without_or_with_failed_reading():
    (s,) = inventory(sept06_frames(), readings=None).sensors
    assert s.adu_scale is None and s.effective_gain is None
    failed = {SENSOR: AduScaleReading("light/a.fits", error="pixels illisibles")}
    inv = inventory(sept06_frames(), readings=failed)
    assert inv.sensors[0].adu_scale is None
    (a,) = [a for a in inv.anomalies if a.code == "adu_scale_failed"]
    assert a.files == ["light/a.fits"] and "pixels illisibles" in a.message


def test_adu_scale_samples_prefer_the_first_light():
    frames = usable_frames(sept06_frames())
    assert adu_scale_samples(frames)[SENSOR].rel_path.startswith("light/H/SESSION_sept06/2021-09-06_23-04-00")
    darks = [f for f in frames if f.imagetyp == DARK]
    assert adu_scale_samples(darks)[SENSOR].imagetyp == DARK


# --------------------------------------------------------------------------- properties of build_sets


@given(st.permutations(sept06_frames()), st.floats(0.5, 48))
def test_build_sets_is_a_partition_independent_of_order(frames, gap_hours):
    for kind in (DARK, FLAT, DARKFLAT):
        members = [f for f in frames if f.imagetyp == kind]
        sets = build_sets(members, kind, gap_hours)
        assert sorted(f.rel_path for s in sets for f in s.frames) == sorted(f.rel_path for f in members)
        for s in sets:
            assert {f.exptime for f in s.frames} == {s.exptime} and {f.imagetyp for f in s.frames} == {kind}
            if kind == FLAT:
                assert {f.filter for f in s.frames} == {s.filter}
        reference = build_sets(sorted(members, key=lambda f: f.rel_path), kind, gap_hours)
        assert [sorted(f.rel_path for f in s.frames) for s in sets] == [
            sorted(f.rel_path for f in s.frames) for s in reference]
