"""astrometry.md and the astrometry table of qa.md."""

from varstar.adapters.reporting.astrometry_markdown import AstrometryReportAdapter
from varstar.adapters.reporting.qa_markdown import qa_markdown
from varstar.domain.qa.model import QAItem


def summary():
    ok = {"raw": "/raw/a.fits", "ok": True, "reason": None, "n_stars": 300, "n_matched": 290, "rms_arcsec": 0.28,
          "orientation_deg": 180.6, "segment": 1, "time": "2021-09-06T23:00:00"}
    bad = {"raw": "/raw/b.fits", "ok": False, "reason": "nuages", "n_stars": 3, "n_matched": 0, "time": "2021-09-06T23:10:00"}
    row = {"group": "g1", "label": "sept06 / H", "n": 2, "n_solved": 1, "rms_median_arcsec": 0.28,
           "scale_median_arcsec": 3.4839, "orientation_median_deg": 180.6, "segments": [1]}
    return {"solver": "interne v1", "catalog": "Gaia DR3 (cache)", "n_images": 2, "n_solved": 1,
            "groups": {"g1": {"images": {"/raw/a.fits": ok, "/raw/b.fits": bad}}}, "rows": [row],
            "errors": {"g2": "images calibrées absentes"}}, row


def test_astrometry_markdown():
    s, _ = summary()
    md = AstrometryReportAdapter().markdown(s, [QAItem("A1", "warning", "sept06 / H : 1/2 image(s) non résolue(s)")])
    assert "Images résolues : 1 / 2" in md and "| sept06 / H | 2 | 1 | 0.28 | 3.4839 | 180.6 | 1 |" in md
    assert "| `b.fits` | **non** | 3 | 0 | — | — | — | nuages |" in md
    assert "A1 — avertissement" in md and "g2 : images calibrées absentes" in md


def test_qa_markdown_astrometry_stage_and_table():
    _, row = summary()
    empty = {**row, "n_solved": 0, "rms_median_arcsec": None, "scale_median_arcsec": None,
             "orientation_median_deg": None, "segments": []}
    md = qa_markdown({"updated": "x", "astrometry": [QAItem("A4", "info", "2 segments").to_dict()],
                      "tables": {"astrometry": [row, empty]}})
    assert "## 4. Astrométrie" in md and "A4 — info" in md
    assert "| sept06 / H | 2 | 0 | — | — | — | — |" in md
