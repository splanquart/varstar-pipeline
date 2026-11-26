"""Markdown rendering (French) of masters.md, compared with a snapshot."""

from varstar.adapters.reporting.masters_markdown import MastersReportAdapter
from varstar.domain.qa.model import QAItem

CORNERS = {"haut_gauche": 0.4, "haut_droit": 0.1, "bas_gauche": 6.2, "bas_droit": 0.0}
PLAN = {
    "created": "2026-11-26T09:00:00", "library": "/lib", "siril_version": "1.4.5",
    "groups": {
        "2021-09-06_H_300s": {
            "key": "2021-09-06_H_300s", "session": "2021-09-06", "label": "s1", "filter": "H", "exptime": 300.0,
            "level": "A", "reasons": [], "files": ["/raw/a.fits"],
            "dark": {"id": "d1", "path": "/lib/d1", "mode": "direct", "n": 8, "exptime": 300.0},
            "flat": {"id": "f1", "path": "/lib/f1", "n": 5}, "darkflat": {"id": "df1", "path": "/lib/df1",
                                                                           "exptime": 1.13},
            "bpm": "/w/masks/bpm_d1_f1.fits", "ampglow": "/w/masks/ampglow_d1.fits"},
        "2021-09-10_S_300s": {
            "key": "2021-09-10_S_300s", "session": "2021-09-10", "label": None, "filter": "S", "exptime": 300.0,
            "level": "B", "reasons": ["dark mis à l'échelle"], "files": ["/raw/b.fits"],
            "dark": {"id": "d2", "path": "/lib/d2", "mode": "scaled", "n": 5, "exptime": 180.0, "offset_adu": 500.0,
                     "offset_master": "df1"},
            "flat": None, "darkflat": None, "bpm": "/w/masks/bpm_d2_noflat.fits", "ampglow": None}},
    "masters": {
        "d1": {"kind": "dark", "label": "DARK_g120_ox_b2_300s_T-12", "n": 8, "rejection": "sigma clipping",
               "median_adu": 516.4, "sigma_single_frame_adu": 12.31, "n_unstable": 42, "level_min_frac": None,
               "pixinsight": {"pixinsight": "masterDark_EXPOSURE-300.00s.xisf", "flipped": True,
                              "median_ours_adu": 516.4, "median_pi_adu": 515.9, "median_diff_adu": 0.5,
                              "robust_sigma_diff_adu": 1.2},
               "consistency": [{"folder": "dark/a", "n": 4, "median_adu": 516.0, "robust_sigma_adu": 12.0,
                                "corner_excess_adu": CORNERS, "mean_ccd_temp": -12.0},
                               {"folder": "dark/b", "n": 4, "median_adu": 519.5, "robust_sigma_adu": 12.4,
                                "corner_excess_adu": CORNERS, "mean_ccd_temp": -11.9}]},
        "f1": {"kind": "flat", "label": "FLAT_g120_ox_b2_1.13s_H_20210907", "n": 5, "rejection": "rejet par percentile",
               "median_adu": 30000.0, "level_min_frac": 0.45, "level_max_frac": 0.47,
               "pixinsight": {"pixinsight": "masterFlat_FILTER-H.xisf", "error": "image XISF compressée"}},
        "df1": {"kind": "darkflat", "label": "DARKFLAT_g120_ox_b2_1.13s", "n": 3, "rejection": "rejet par percentile",
                "median_adu": 500.2}},
    "warnings": ["2021-09-10_S_300s : dark mis à l'échelle depuis 180 s (offset estimé 500.0 ADU) — niveau B"],
}
MASKS = {"/w/masks/bpm_d1_f1.fits": {"counts": {"hot": 120, "very_hot": 3, "dead": 1}, "info": {}},
         "/w/masks/ampglow_d1.fits": {"counts": {}, "info": {"fraction": 0.041, "max_excess_adu": 37.6}}}


def test_masters_markdown_snapshot(snapshot):
    items = [QAItem("H9", "info", "s1 / H : 8 darks seulement (< 15) pour le master")]
    snapshot("masters.md", MastersReportAdapter().markdown(PLAN, items, MASKS))


def test_without_qa_report_nor_reference_masters():
    plan = {**PLAN, "masters": {k: {kk: vv for kk, vv in m.items() if kk != "pixinsight"}
                                for k, m in PLAN["masters"].items()}, "warnings": []}
    md = MastersReportAdapter().markdown(plan, None, {})
    assert "## Contrôles qualité" not in md and "Aucun master PixInsight correspondant trouvé." in md
    assert "## Avertissements" not in md and "| — | — |" in md
