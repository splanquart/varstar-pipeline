"""The normalisation and diff used by the golden tests and the sept06 acceptance check."""

import datetime as dt

from compare_outputs import diff, normalize, normalize_text

ID = "d02977b81eecfba6"
ALIASES = {ID: "<DARK@2021-09-07T02:00>"}


def test_normalize_drops_volatile_keys_and_replaces_paths():
    data = {"created": "x", "siril_version": "1.4.5", "root": "/tmp/abc/raw/light", "n": 3}
    assert normalize(data, {"/tmp/abc/raw": "<RAW>", "/tmp/abc": "<TMP>"}) == {"root": "<RAW>/light", "n": 3}


def test_normalize_aliases_master_ids_even_next_to_underscores():
    s = f"/lib/darks/DARK_300s_{ID}/master.fits bpm_{ID}_noflat_372d1e34.fits ({ID[:8]})"
    assert normalize_text(s, aliases=ALIASES) == (
        "/lib/darks/DARK_300s_<DARK@2021-09-07T02:00>/master.fits "
        "bpm_<DARK@2021-09-07T02:00>_noflat_372d1e34.fits (<DARK@2021-09-07T02:00>)")
    # Unknown 16-hex ids are masked, 8-hex config hashes are kept.
    assert normalize_text("x_0123456789abcdef_372d1e34") == "x_<id>_372d1e34"
    # Ids are aliased in dict keys too.
    assert normalize({ID: 1}, aliases=ALIASES) == {"<DARK@2021-09-07T02:00>": 1}


def test_normalize_masks_run_times_but_keeps_observation_dates():
    now = dt.datetime.now()
    s = f"généré le {now:%Y-%m-%dT%H:%M:%S} / {now:%d/%m/%Y %H:%M} ; DATE-OBS 2021-09-06T23:00:00"
    assert normalize_text(s) == "généré le <NOW> / <NOW> ; DATE-OBS 2021-09-06T23:00:00"


def test_diff_uses_numeric_tolerances_and_reports_structure():
    assert diff({"a": 1.0, "b": [1, 2]}, {"a": 1.0 + 1e-9, "b": [1, 2]}) == []
    assert diff({"a": 1.0}, {"a": 1.1}) == [".a : 1.0 → 1.1"]
    assert diff({"a": 1}, {"b": 1}) == [".a : clé disparue", ".b : clé nouvelle"]
    assert diff([1, 2], [1]) == ["<racine> : 2 éléments → 1"]
    assert diff({"a": True}, {"a": 1}) == [".a : True → 1"]
    assert diff(float("nan"), float("nan")) == []
