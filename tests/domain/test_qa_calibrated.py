"""Post-calibration checks (C*) from per-frame statistics in memory."""

from builders import make_settings
from varstar.domain.qa.calibrated import ampglow_residual_items, calibrated_group_checks
from varstar.domain.qa.model import QAItem, report_items, severity_counts, sort_items

GROUP = {"label": "sept06", "session": "2021-09-06", "filter": "H", "dark": {"id": "d1"}}


def frame(bg=50.0, sigma=5.0, nan=0.0, zero=0.0, resid=(), raw="light/a.fits"):
    return {"file": "/w/calibrated/a.fit", "raw": raw, "background_adu": bg, "sigma_adu": sigma, "frac_nan": nan,
            "frac_zero": zero, "frac_low_5sigma": 0.0, "ampglow_zone_resid_adu": list(resid)}


def test_clean_group_gives_a_row_and_no_item():
    items, row = calibrated_group_checks("g1", GROUP, [frame(48, resid=[1.0]), frame(52, resid=[3.0])],
                                         make_settings())
    assert items == []
    assert row == {"group": "g1", "label": "sept06 / H", "n": 2, "background_min_adu": 48,
                   "background_median_adu": 50.0, "sigma_median_adu": 5.0, "frac_nan_max": 0.0,
                   "frac_zero_max": 0.0, "ampglow_zone_resid_adu": [2.0], "ampglow_excess_median_adu": 2.0,
                   "dark_id": "d1", "session": "2021-09-06"}


def test_negative_background_and_invalid_pixels_are_c3():
    per = [frame(-4.0, raw="light/a.fits"), frame(20.0, nan=0.001, raw="light/b.fits")]
    items, row = calibrated_group_checks("g1", {**GROUP, "label": None, "dark": None}, per,
                                         make_settings())
    assert [(i.check, i.severity) for i in items] == [("C3", "error"), ("C3", "warning")]
    assert items[0].files == ["light/a.fits"] and all(i.data["group"] == "g1" for i in items)
    assert row["label"] == "2021-09-06 / H" and row["dark_id"] is None


def test_empty_group_has_no_row():
    assert calibrated_group_checks("g1", GROUP, [], make_settings()) == ([], None)


def test_ampglow_residual_needs_all_filters():
    rows = [{"group": "g1", "label": "s / H", "session": "s", "dark_id": "d", "ampglow_zone_resid_adu": [-24.5, 1.0]},
            {"group": "g2", "label": "s / S", "session": "s", "dark_id": "d", "ampglow_zone_resid_adu": [-5.3, 0.5]}]
    items = ampglow_residual_items(rows)
    assert [(i.check, i.severity) for i in items] == [("C1", "info")]
    rows[1]["ampglow_zone_resid_adu"] = [-20.0, 0.5]
    (c1,) = ampglow_residual_items(rows)
    assert c1.severity == "warning" and "sur-corrigé" in c1.message and c1.data["zone"] == 1
    assert "un seul filtre" in ampglow_residual_items(rows[:1])[0].message


def test_report_items_and_counts():
    report = {"header": [QAItem("H9", "info", "a").to_dict()], "raw": [QAItem("P1", "warning", "b").to_dict()],
              "calibrated": [QAItem("C3", "error", "c").to_dict()]}
    assert [i.check for i in report_items(report)] == ["C3", "P1", "H9"]
    assert [i.check for i in report_items(report, ("header",))] == ["H9"]
    assert severity_counts(report_items(report)) == {"error": 1, "warning": 1, "info": 1}
    assert [i.check for i in sort_items([QAItem("P2", "info", ""), QAItem("H1", "info", "")])] == ["H1", "P2"]
