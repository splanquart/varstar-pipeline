"""Star catalogues: local CSV, Gaia DR3 over TAP (network faked, except under the « network » marker)."""

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
