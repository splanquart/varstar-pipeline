"""Star catalogues: local CSV, Gaia DR3 over TAP (network faked, except under the « network » marker)."""

from pathlib import Path

import numpy as np
import pytest
import pyvo
import requests

from varstar.adapters.star_catalog import CsvStarCatalog, GaiaTapCatalog, parse_csv, table_to_csv
from varstar.domain.calibration import PipelineError

GAIA_CSV = ("source_id,ra,dec,phot_g_mean_mag\n"
            "1,60.44,36.11,9.5\n2,60.50,36.10,8.0\n3,60.44,36.60,10.0\n4,60.45,36.12,14.5\n5,61.40,36.11,7.0\n")


def gaia_table(text: str = GAIA_CSV) -> dict:
    """What pyvo's ``to_table()`` gives for the Gaia query: one column per field."""
    st = parse_csv(text)
    return {"source_id": np.array([int(i) for i in st.ids]), "ra": st.ra, "dec": st.dec, "phot_g_mean_mag": st.mag}


def test_table_to_csv_skips_stars_without_magnitude():
    t = gaia_table()
    t["phot_g_mean_mag"] = np.ma.masked_array(t["phot_g_mean_mag"], mask=[0, 0, 1, 0, 0])
    st = parse_csv(table_to_csv(t))
    assert st.n == 4 and "3" not in st.ids
    with pytest.raises(KeyError):
        table_to_csv({"ra": [1.0]})


def test_parse_csv_variants():
    st = parse_csv(GAIA_CSV)
    assert st.n == 5 and st.ids[0] == "1" and st.mag[1] == 8.0
    st = parse_csv("RA,Dec,mag\n1,2,3\n4,5,\n")          # case-insensitive, empty magnitude dropped
    assert st.n == 1 and st.ids == ()
    assert parse_csv("").n == 0
    with pytest.raises(ValueError, match="ra et dec"):
        parse_csv("x,y,mag\n1,2,3\n")
    with pytest.raises(ValueError, match="magnitude"):
        parse_csv("ra,dec,flux\n1,2,3\n")


def test_csv_catalog_filters_and_sorts(tmp_path):
    p = tmp_path / "cat.csv"
    p.write_text(GAIA_CSV)
    cat = CsvStarCatalog(p)
    st = cat.cone(60.44, 36.11, 0.3, 13.0)
    assert list(st.ids) == ["2", "1"] and st.source == "fichier cat.csv"     # brightest first, in the cone
    with pytest.raises(PipelineError, match="illisible"):
        CsvStarCatalog(tmp_path / "absent.csv").cone(0, 0, 1, 13)


def test_gaia_downloads_once_then_uses_the_cache(tmp_path):
    calls = []

    def fetch(url, query):
        calls.append((url, query))
        return gaia_table(), False

    cat = GaiaTapCatalog(tmp_path / "catalogs", ["https://tap.example/tap"], fetch=fetch)
    st = cat.cone(60.44, 36.11, 0.3, 13.0)
    assert list(st.ids) == ["2", "1"] and "tap.example" in st.source
    assert "CIRCLE('ICRS', 60.440000, 36.110000, 0.3000)" in calls[0][1] and "< 13.00" in calls[0][1]
    # A smaller cone inside the cached one, not deeper: answered from the cache.
    again = GaiaTapCatalog(tmp_path / "catalogs", ["https://tap.example/tap"], fetch=fetch)
    st2 = again.cone(60.45, 36.11, 0.2, 12.0)
    assert len(calls) == 1 and "cache" in st2.source and list(st2.ids) == ["2", "1"]
    # Deeper or larger: a new download.
    again.cone(60.44, 36.11, 0.3, 15.0)
    again.cone(60.44, 36.11, 0.9, 13.0)
    assert len(calls) == 3
    assert len(list((tmp_path / "catalogs").glob("gaia_dr3_*.csv"))) == 3


def pyvo_error(kind: str) -> Exception:
    """The exceptions as pyvo 1.10 really raises them from ``run_sync`` (checked against the real services):
    connection failures and timeouts come wrapped in a DALFormatError, never as requests exceptions."""
    if kind == "down":
        return pyvo.dal.DALFormatError(requests.ConnectionError("Failed to resolve 'down'"), "https://down/tap/sync")
    if kind == "slow":
        return pyvo.dal.DALFormatError(requests.ReadTimeout("read timed out"), "https://slow/tap/sync")
    if kind == "html":
        return pyvo.dal.DALFormatError(ValueError("1:0: syntax error"), "https://html/tap/sync")
    if kind == "http":
        return pyvo.dal.DALServiceError("500 Server Error: Internal Server Error", 500, url="https://http/tap/sync")
    return pyvo.dal.DALQueryError("unknown table gaiadr3.gaia_source")


def test_gaia_falls_back_and_explains_failures(tmp_path):
    answers = {f"https://{k}/tap": pyvo_error(k) for k in ("down", "slow", "html", "http", "adql")}
    answers.update({"https://full/tap": (gaia_table(), True), "https://cols/tap": ({"ra": [1.0]}, False),
                    "https://ok/tap": (gaia_table(), False)})

    def fetch(url, query):
        a = answers[url]
        if isinstance(a, Exception):
            raise a
        return a

    order = [f"https://{k}/tap" for k in ("down", "slow", "html", "http", "adql", "full", "cols")]
    cat = GaiaTapCatalog(tmp_path, order + ["https://ok/tap"], timeout_s=30, fetch=fetch)
    st = cat.cone(60.44, 36.11, 0.3, 13.0)
    assert st.n == 2 and "ok" in st.source
    bad = GaiaTapCatalog(tmp_path / "other", order, timeout_s=30, fetch=fetch)
    with pytest.raises(PipelineError) as exc:
        bad.cone(60.44, 36.11, 0.3, 13.0)
    msg = str(exc.value)
    for part in ("down : injoignable", "slow : pas de réponse en 30 s", "html : réponse inattendue",
                 "http : 500 Server Error", "adql : unknown table", "full : résultat tronqué",
                 "cols : colonnes inattendues", "astrometry.catalog"):
        assert part in msg, part


@pytest.mark.network
def test_real_unreachable_service_is_reported_as_such(tmp_path):
    """Against the real pyvo: a host that does not resolve is « injoignable », not « réponse inattendue »."""
    cat = GaiaTapCatalog(tmp_path, ["https://nonexistent.invalid/tap"], timeout_s=10)
    with pytest.raises(PipelineError, match="nonexistent.invalid : injoignable"):
        cat.cone(60.44, 36.11, 0.1, 10.0)


def test_gaia_ignores_broken_cache_entries(tmp_path):
    (tmp_path / "gaia_dr3_x.json").write_text("{not json")
    (tmp_path / "gaia_dr3_y.json").write_text('{"version": 0}')
    cat = GaiaTapCatalog(tmp_path, ["https://ok/tap"], fetch=lambda u, q: (gaia_table(), False))
    assert cat.cone(60.44, 36.11, 0.3, 13.0).n == 2


@pytest.mark.network
def test_gaia_real_service(tmp_path):
    cat = GaiaTapCatalog(tmp_path, ["https://gaia.ari.uni-heidelberg.de/tap", "https://gea.esac.esa.int/tap-server/tap"])
    st = cat.cone(60.44, 36.11, 0.5, 12.0)
    assert st.n > 10 and np.all(np.diff(st.mag) >= 0)


# --------------------------------------------------------------------------- robustness (review of PR #13)


def ok_fetch(calls=None):
    def fetch(url, query):
        if calls is not None:
            calls.append(url)
        return gaia_table(), False
    return fetch


@pytest.mark.parametrize("meta", ["[]", '{"version": 1}', '{"version": 1, "ra": "x", "dec": 36.11, '
                                  '"radius_deg": 1, "mag_limit": 13}', '{"version": 1, "ra": NaN, "dec": 36.11, '
                                  '"radius_deg": 1, "mag_limit": 13}', "\xff\xfe"])
def test_malformed_cache_metadata_is_skipped(tmp_path, meta, caplog):
    (tmp_path / "gaia_dr3_bad.json").write_bytes(meta.encode("latin-1"))
    calls = []
    st = GaiaTapCatalog(tmp_path, ["https://ok/tap"], fetch=ok_fetch(calls)).cone(60.44, 36.11, 0.3, 13.0)
    assert st.n == 2 and calls == ["https://ok/tap"]           # downloaded again
    assert "gaia_dr3_bad.json illisible" in caplog.text


@pytest.mark.parametrize("content", [None, "ra,dec\n1,2\n", "ra,dec,mag\nabc,2,3\n", "ra,dec,mag\n1\n"])
def test_corrupt_or_missing_cache_csv_is_skipped(tmp_path, content, caplog):
    cat = GaiaTapCatalog(tmp_path, ["https://ok/tap"], fetch=ok_fetch())
    cat.cone(60.44, 36.11, 0.3, 13.0)                         # creates a valid entry
    (csv_path,) = tmp_path.glob("gaia_dr3_*.csv")
    if content is None:
        csv_path.unlink()
    else:
        csv_path.write_text(content)
    calls = []
    st = GaiaTapCatalog(tmp_path, ["https://ok/tap"], fetch=ok_fetch(calls)).cone(60.44, 36.11, 0.3, 13.0)
    assert st.n == 2 and calls == ["https://ok/tap"]
    assert "ignoré" in caplog.text


def test_cache_write_failure_keeps_the_downloaded_stars(tmp_path, caplog):
    blocker = tmp_path / "catalogs"
    blocker.write_text("not a directory")                    # mkdir fails: OSError
    st = GaiaTapCatalog(blocker, ["https://ok/tap"], fetch=ok_fetch()).cone(60.44, 36.11, 0.3, 13.0)
    assert st.n == 2 and "catalogue utilisé sans cache" in caplog.text


def test_cache_files_are_written_atomically_with_unique_temporaries(tmp_path, monkeypatch):
    import varstar.adapters.star_catalog as sc
    seen = []
    real_mkstemp = sc.tempfile.mkstemp

    def spy(**kw):
        fd, name = real_mkstemp(**kw)
        seen.append(name)
        return fd, name

    monkeypatch.setattr(sc.tempfile, "mkstemp", spy)
    GaiaTapCatalog(tmp_path, ["https://ok/tap"], fetch=ok_fetch()).cone(60.44, 36.11, 0.3, 13.0)
    assert len(seen) == 2 and len(set(seen)) == 2              # CSV, then metadata
    assert seen[0].endswith(".tmp") and ".csv." in seen[0] and ".json." in seen[1]
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".csv", ".json"]   # no temporary left
    # A failed write leaves no temporary file behind and changes nothing.
    monkeypatch.setattr(sc.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("disque plein")))
    with pytest.raises(OSError):
        sc.atomic_write(tmp_path / "x.csv", "data")
    assert not list(tmp_path.glob(".x.csv.*")) and not (tmp_path / "x.csv").exists()


def test_table_to_csv_requires_every_column_of_a_star():
    t = gaia_table()
    t["ra"] = np.ma.masked_array(t["ra"], mask=[0, 1, 0, 0, 0])
    t["dec"] = np.array([36.11, 36.10, np.nan, 36.12, 36.11])
    t["source_id"] = np.ma.masked_array(t["source_id"], mask=[0, 0, 0, 1, 0])
    st = parse_csv(table_to_csv(t))
    assert list(st.ids) == ["1", "5"]
    t = gaia_table()
    t["dec"] = t["dec"][:3]
    with pytest.raises(ValueError, match="longueurs différentes"):
        table_to_csv(t)
    # Through cone(): a table with columns of different lengths is « colonnes inattendues », next service tried.
    cat = GaiaTapCatalog(Path("/nonexistent/cache"), ["https://bad/tap"], fetch=lambda u, q: (t, False))
    with pytest.raises(PipelineError, match="bad : colonnes inattendues"):
        cat.cone(60.44, 36.11, 0.3, 13.0)


def test_parse_csv_drops_non_finite_values_and_names_bad_lines(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="varstar")
    st = parse_csv("ra,dec,mag\n60.44,36.11,9.5\nnan,36.1,8\n60.5,inf,8\n60.5,36.1,-inf\n60.5,36.2,8\n")
    assert st.n == 2 and list(st.mag) == [9.5, 8.0]
    assert "3 ligne(s)" in caplog.text
    with pytest.raises(ValueError, match="ligne 3 illisible"):
        parse_csv("ra,dec,mag\n1,2,3\nabc,2,3\n")
    with pytest.raises(ValueError, match="ligne 2 illisible"):
        parse_csv("ra,dec,mag\n1,2\n")


def test_pyvo_wraps_connection_errors_and_timeouts(tmp_path, monkeypatch):
    """Pins the pyvo behaviour the error messages rely on (see ``tap_problem``): with the real pyvo, a
    connection error or a timeout raised by requests comes out of ``run_sync`` as a DALFormatError whose
    ``cause`` is the requests exception — never as a requests exception. If a pyvo upgrade changes this,
    this test fails and ``cone()`` must be adapted."""
    for error, expected in ((requests.ConnectionError("down"), "injoignable"),
                            (requests.ReadTimeout("slow"), "pas de réponse en 7 s")):
        monkeypatch.setattr(requests.Session, "request", lambda *a, _e=error, **k: (_ for _ in ()).throw(_e))
        cat = GaiaTapCatalog(tmp_path / type(error).__name__, ["https://example.org/tap"], timeout_s=7)
        with pytest.raises(pyvo.dal.DALFormatError) as exc:
            cat._tap("https://example.org/tap", "SELECT 1")
        assert exc.value.cause is error
        with pytest.raises(PipelineError, match=f"example.org : {expected}"):
            cat.cone(60.44, 36.11, 0.3, 13.0)
