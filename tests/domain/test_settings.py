"""Interpretation of matching.same_optical_setup (periods where flats can be shared)."""

import datetime as dt

import pytest

from varstar.domain.settings import optical_setup_periods


def cfg(periods):
    return periods


def test_periods_accept_strings_dates_and_datetimes_and_reorder():
    periods = optical_setup_periods(cfg([["2021-09-02", "2021-09-07"],
                                         [dt.datetime(2021, 9, 20, 12), dt.date(2021, 9, 12)]]))
    assert periods == [(dt.date(2021, 9, 2), dt.date(2021, 9, 7)), (dt.date(2021, 9, 12), dt.date(2021, 9, 20))]


@pytest.mark.parametrize("raw", [None, []])
def test_no_period(raw):
    assert optical_setup_periods(cfg(raw)) == []


@pytest.mark.parametrize("bad", [["2021-09-02"], "2021-09-02", ["2021-09-02", "2021-09-03", "2021-09-04"]])
def test_invalid_period_is_reported_in_french(bad):
    with pytest.raises(ValueError, match="période invalide"):
        optical_setup_periods(cfg([bad]))


def test_astrometry_engine_is_checked():
    from builders import make_settings
    from varstar.domain.settings import SettingsError
    assert make_settings().astrometry.engine == "internal"
    assert make_settings().astrometry.tap_urls[0].startswith("https://gaia.ari")
    with pytest.raises(SettingsError, match="astrometry.engine = 'siril'.*D17"):
        make_settings(astrometry={"engine": "siril"})


def test_rotation_answers_are_parsed_and_checked():
    import datetime as dt
    from builders import make_settings
    from varstar.domain.settings import SettingsError
    a = make_settings().astrometry
    assert a.whole_train_rotation is None and a.whole_train_rotation_nights == ()
    a = make_settings(astrometry={"whole_train_rotation_nights": {dt.date(2021, 9, 2): True, "2021-08-31": False}}).astrometry
    assert a.whole_train_rotation_nights == ((dt.date(2021, 8, 31), False), (dt.date(2021, 9, 2), True))
    with pytest.raises(SettingsError, match="true ou false"):
        make_settings(astrometry={"whole_train_rotation_nights": {"2021-09-02": "oui"}})
    with pytest.raises(SettingsError, match="date invalide"):
        make_settings(astrometry={"whole_train_rotation_nights": {"hier": True}})
    with pytest.raises(SettingsError, match="AAAA-MM-JJ"):
        make_settings(astrometry={"whole_train_rotation_nights": [True]})
