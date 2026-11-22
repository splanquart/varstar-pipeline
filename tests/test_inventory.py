import datetime as dt
import shutil

from varstar.config import load_config
from varstar.discovery import discover
from varstar.inventory import build_inventory, darks_todo, worst

from fitsgen import series, write_frame


def levels(inv):
    return {(s.isoformat(), f): lv for (s, f), lv in inv.levels_by_session_filter().items()}


def group(inv, filt, session=None):
    gs = [g for g in inv.light_groups if g.filter == filt and (session is None or g.session.isoformat() == session)]
    assert len(gs) == 1
    return gs[0]


def test_worst():
    assert worst("A", "B") == "B"
    assert worst("A", None, "C") == "C"
    assert worst() == "C"


def test_discovery_ignores(tmp_path, cfg):
    write_frame(tmp_path / "light/a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "removed/b.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "light/S/SESSION_x/removed/c.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "output/master/d.fit", "DARK", 300, "2021-09-06T23:00:00")
    write_frame(tmp_path / "proj.pxiproject/e.fits", "DARK", 300, "2021-09-06T23:00:00")
    write_frame(tmp_path / "light/._a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    (tmp_path / "light/img.xisf").write_text("x")
    (tmp_path / "bias/SESSION_1").mkdir(parents=True)
    res = discover(tmp_path, cfg)
    assert [p.name for p in res.fits_files] == ["a.fits"]
    assert "bias/SESSION_1" in res.empty_dirs
    assert res.other_extensions[".xisf"] == 1
    cfg["inventory"]["include_removed"] = True
    assert sorted(p.name for p in discover(tmp_path, cfg).fits_files) == ["a.fits", "b.fits", "c.fits"]


def test_complete_session_is_level_a(complete_session, cfg):
    inv = build_inventory(complete_session, cfg)
    assert levels(inv) == {("2021-09-06", "H"): "A", ("2021-09-06", "S"): "A"}
    g = group(inv, "H")
    assert g.n == 3 and g.dark.n_frames == 4 and g.flat.sets[0].exptime == 1.13
    assert g.darkflat.sets[0].exptime == 1.13
    assert darks_todo(inv) == []


def test_header_wins_over_folder(complete_session, cfg):
    # A 180 s dark misfiled in a 300s folder, and an H light stored in an S folder.
    write_frame(complete_session / "dark/darklight/300s/x.fits", "DARK", 180, "2021-09-07T03:00:00")
    write_frame(complete_session / "light/S/SESSION_sept06/y.fits", "LIGHT", 300, "2021-09-07T01:00:00", filt="H")
    inv = build_inventory(complete_session, cfg)
    codes = {a.code for a in inv.anomalies}
    assert {"folder_exptime", "folder_filter"} <= codes
    assert group(inv, "H").n == 4
    assert any(s.exptime == 180 for s in inv.calibration_sets["DARK"])


def test_missing_exact_darkflat_gives_b(complete_session, cfg):
    for p in (complete_session / "flat/SESSION_sept06").glob("*_H_*"):
        p.unlink()
    series(complete_session, "flat/SESSION_sept06", "FLAT", 1.61, "2021-09-07T13:21:00", 3, step_s=2, filt="H")
    series(complete_session, "dark/darkflat", "DARKFLAT", 1.69, "2021-09-07T13:40:00", 3, step_s=2, prefix="")
    inv = build_inventory(complete_session, cfg)
    g = group(inv, "H")
    assert g.level == "B" and g.darkflat.level == "B" and g.darkflat.sets[0].exptime == 1.69
    todo = darks_todo(inv)
    assert [(t.kind, t.exptime) for t in todo] == [("darkflat", 1.61)]


def test_darkflat_tolerance(complete_session, cfg):
    for p in (complete_session / "dark/darkflat").glob("*1.13s*"):
        p.unlink()
    series(complete_session, "dark/darkflat", "DARKFLAT", 1.16, "2021-09-08T18:25:00", 3, step_s=2, prefix="")
    assert group(build_inventory(complete_session, cfg), "H").darkflat.level == "A"


def test_no_darkflat_at_all_makes_flats_unusable(complete_session, cfg):
    shutil.rmtree(complete_session / "dark/darkflat")
    shutil.rmtree(complete_session / "dark/darklight")
    inv = build_inventory(complete_session, cfg)
    g = group(inv, "H")
    assert g.flat.level == "C" and g.level == "C"


def test_darks_from_another_exposure_are_scaled_b(complete_session, cfg):
    shutil.rmtree(complete_session / "dark/darklight")
    series(complete_session, "dark/darklight/180s", "DARK", 180, "2021-09-07T02:00:00", 3, step_s=200, prefix="")
    series(complete_session, "dark/darklight/600s", "DARK", 600, "2021-09-07T03:00:00", 3, step_s=610, prefix="")
    inv = build_inventory(complete_session, cfg)
    g = group(inv, "H")
    assert g.dark.level == "B" and g.dark.sets[0].exptime == 180  # shorter darks preferred
    assert "mise à l'échelle" in g.dark.reasons[0]
    assert [(t.kind, t.exptime) for t in darks_todo(inv)] == [("dark", 300.0)]
    cfg["matching"]["allow_dark_scaling"] = False
    assert group(build_inventory(complete_session, cfg), "H").dark.level == "C"


def test_dark_temperature_mismatch_is_b(complete_session, cfg):
    shutil.rmtree(complete_session / "dark/darklight")
    series(complete_session, "dark/darklight/300s", "DARK", 300, "2021-09-07T02:00:00", 3, step_s=310,
           prefix="", temp=-5.0)
    g = group(build_inventory(complete_session, cfg), "H")
    assert g.dark.level == "B" and "ΔT" in g.dark.reasons[0]


def test_sensor_key_must_match(complete_session, cfg):
    shutil.rmtree(complete_session / "dark/darklight")
    series(complete_session, "dark/darklight/300s", "DARK", 300, "2021-09-07T02:00:00", 3, step_s=310,
           prefix="", gain=200)
    inv = build_inventory(complete_session, cfg)
    assert group(inv, "H").dark.level == "C"
    assert any(a.code == "several_sensors" for a in inv.anomalies)


def _add_session_without_flats(root):
    series(root, "light/H/SESSION_sept02", "LIGHT", 300, "2021-09-02T23:28:00", 3, step_s=600, filt="H")


def test_session_without_flats_is_c_then_b_with_same_setup(complete_session):
    _add_session_without_flats(complete_session)
    cfg = load_config()
    g = group(build_inventory(complete_session, cfg), "H", "2021-09-02")
    assert g.flat.level == "C" and g.level == "C"
    assert "same_optical_setup" in g.flat.reasons[0]

    cfg = load_config(overrides={"matching": {"same_optical_setup": [["2021-09-02", "2021-09-07"]]}})
    g = group(build_inventory(complete_session, cfg), "H", "2021-09-02")
    assert g.flat.level == "B" and g.level == "B"
    assert g.flat.sets[0].start.date() == dt.date(2021, 9, 7)

    cfg = load_config(overrides={"matching": {"same_optical_setup": [["2021-09-02", "2021-09-07"]],
                                              "max_flat_age_days": 3}})
    assert group(build_inventory(complete_session, cfg), "H", "2021-09-02").flat.level == "C"


def test_flats_in_same_session_folder_but_days_before(tmp_path):
    """Question Q3: flats stored in SESSION_aout31 but taken on 29/08."""
    root = tmp_path / "Y"
    series(root, "light/H/SESSION_aout31", "LIGHT", 300, "2021-08-31T23:28:00", 2, step_s=600, filt="H")
    series(root, "flat/SESSION_aout31", "FLAT", 1.13, "2021-08-29T16:09:00", 3, step_s=2, filt="H")
    series(root, "dark/300s", "DARK", 300, "2021-09-01T02:18:00", 3, step_s=310, prefix="")
    series(root, "dark/darkflat", "DARKFLAT", 1.13, "2021-09-08T18:25:00", 3, step_s=2, prefix="")
    cfg = load_config()
    g = group(build_inventory(root, cfg), "H")
    assert g.level == "A" and "Q3" in g.flat.reasons[0] and "avant" in g.flat.reasons[0]
    cfg["matching"]["trust_session_folder_flats"] = False
    assert group(build_inventory(root, cfg), "H").flat.level == "C"


def test_flats_of_other_filter_not_used(complete_session, cfg):
    for p in (complete_session / "flat/SESSION_sept06").glob("*_S_*"):
        p.unlink()
    inv = build_inventory(complete_session, cfg)
    assert group(inv, "S").flat.level == "C"
    assert group(inv, "H").flat.level == "A"


def test_offset_missing_warning_once(complete_session, cfg):
    inv = build_inventory(complete_session, cfg)
    assert sum(a.code == "offset_missing" for a in inv.anomalies) == 1


def test_sensor_info_effective_gain(complete_session, cfg):
    inv = build_inventory(complete_session, cfg)
    (s,) = inv.sensors
    assert s.adu_scale == 4
    assert abs(s.effective_gain - 1.00224268436432 / 4) < 1e-9
    assert s.saturation_adu == 65532
    assert abs(s.pixel_scale_arcsec - 3.485) < 0.01


def test_bad_header_frame_excluded(complete_session, cfg):
    write_frame(complete_session / "light/H/z.fits", "WEIRD", 300, "2021-09-06T23:30:00", filt="H")
    inv = build_inventory(complete_session, cfg)
    assert group(inv, "H").n == 3
    assert any(a.code == "bad_header" for a in inv.anomalies)
