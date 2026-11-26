"""Markdown rendering (French) of the quality checks, compared with a snapshot."""

from varstar.adapters.reporting.qa_markdown import QAReportAdapter, items_markdown, summary_line
from varstar.domain.qa.model import QAItem

REPORT = {
    "updated": "2026-11-22T10:00:00",
    "header": [QAItem("H9", "info", "sept06 / H : 5 darks seulement (< 15) pour le master", [],
                      "master dark plus bruité", "faire au moins 15 darks par pose et par nuit").to_dict()],
    "raw": [QAItem("P1", "warning", "darks 300 s, nuit(s) du 06/09 (sept06) : niveau 541.0 ADU contre 516.0 ADU",
                   [f"/raw/dark/b/2021-09-07_02-0{i}-00_Dark.fits" for i in range(5)], "décalage du fond",
                   "fixer l'offset", {"delta_adu": 25.0}).to_dict()],
    "calibrated": [QAItem("C3", "error", "sept06 / H : fond calibré négatif sur 1/3 image(s)",
                          ["light/H/a.fits"]).to_dict()],
    "tables": {
        "dark_levels": [{"set": "DARK-300s-20210902", "kind": "DARK", "exptime": 300.0, "night": "2021-09-02", "n": 5,
                         "level_adu": 516.0, "sigma_adu": 12.0, "mean_ccd_temp": -12.0, "folders": ["dark/a"]},
                        {"set": "DARK-300s-20210907", "kind": "DARK", "exptime": 300.0, "night": "2021-09-06", "n": 5,
                         "level_adu": 541.0, "sigma_adu": 12.0, "mean_ccd_temp": None, "folders": ["dark/b"]}],
        "raw_sky": [{"group": "sept06 / H", "light_median_adu": 600.0, "dark_level_adu": 541.0, "raw_sky_adu": 59.0,
                     "dark_level": "A"}],
        "flat_pairs": [{"filter": "H", "a": "FLAT-H-1", "b": "FLAT-H-2", "a_labels": ["sept06"], "b_labels": [],
                        "dust_pct": 2.1, "shape_pct": 0.4, "noise_pct": 0.1, "verdict": "poussières déplacées"}],
        "calibrated": [{"group": "g1", "label": "sept06 / H", "n": 3, "background_min_adu": -4.0,
                        "background_median_adu": 12.0, "sigma_median_adu": 5.0, "ampglow_excess_median_adu": None,
                        "frac_nan_max": 0.0, "frac_zero_max": 0.0}],
    },
}


def test_qa_markdown_snapshot(snapshot):
    snapshot("qa.md", QAReportAdapter().markdown(REPORT))


def test_empty_stage_and_items_section():
    md = QAReportAdapter().markdown({"updated": "x", "header": []})
    assert "## 1. En-têtes\n\nAucun problème détecté." in md and "## 2." not in md
    items = [QAItem("P1", "warning", "m", advice="a")]
    assert summary_line(items) == "0 erreur(s), 1 avertissement(s), 0 info(s)"
    lines = items_markdown(items, "## QA")
    assert lines[0] == "## QA" and "- **P1 — avertissement** : m" in lines
    assert items_markdown([])[-2] == "Aucun problème détecté."


def test_flat_without_session_label_has_no_empty_parentheses():
    # https://github.com/splanquart/varstar-pipeline/issues/6
    md = QAReportAdapter().markdown(REPORT)
    assert "| H | FLAT-H-1 (sept06) | FLAT-H-2 | 2.10 |" in md and "()" not in md
