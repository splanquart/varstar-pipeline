"""Use case « contrôles qualité » with in-memory fakes (no file, no FITS)."""

import datetime as dt

import numpy as np
import pytest

from builders import inventory_of, make_frame_stats, make_settings, sept06_frames, series_frames, two_nights_frames
from fakes import FakePixelStats, InMemoryArtifactStore, RecordingQAReport
from varstar.application.check import (calibrated_checks, load_report, raw_checks, record_calibrated_checks,
                                       run_checks, save_report)
from varstar.domain.model import FLAT
from varstar.domain.qa.model import QAItem

NOW = dt.datetime(2026, 11, 22, 10, 0, 0)


def dark_levels(f):
    return make_frame_stats(541.0 if f.rel_path.startswith("dark/b") else 516.0 if f.rel_path.startswith("dark/")
                            else 600.0)


@pytest.fixture
def ports():
    return {"store": InMemoryArtifactStore(), "pixels": FakePixelStats(default=dark_levels),
            "report": RecordingQAReport()}


def test_run_checks_publishes_header_and_raw_stages(ports):
    inv = inventory_of(two_nights_frames())
    rep = run_checks(inv, inv.config, now=NOW, **ports)
    store = ports["store"]
    assert set(store.files) == {"qa/qa.json", "qa/qa.md"}
    assert store.read_json("qa/qa.json") == rep and rep["updated"] == "2026-11-22T10:00:00"
    assert [d["check"] for d in rep["raw"]] == ["P1"]
    assert {"dark_levels", "flat_levels", "raw_sky"} <= set(rep["tables"])
    assert ports["report"].rendered[-1] == rep
    assert len(ports["pixels"].measured) == 10 + 6          # every dark + first/middle/last lights


def test_headers_only_reads_no_pixel_and_keeps_the_other_stages(ports):
    inv = inventory_of(two_nights_frames())
    c3, p1 = QAItem("C3", "error", "fond négatif").to_dict(), QAItem("P1", "warning", "offset").to_dict()
    ports["store"].write_json("qa/qa.json", {"calibrated": [c3], "raw": [p1]})
    rep = run_checks(inv, inv.config, raw=False, now=NOW, **ports)
    assert ports["pixels"].measured == []
    assert rep["calibrated"] == [c3] and rep["raw"] == [p1]
    assert "header" in rep


def test_corrupted_report_starts_again(ports):
    ports["store"].write_text("qa/qa.json", "{not json")
    assert load_report(ports["store"]) == {}
    ports["store"].write_json("qa/qa.json", [1, 2])
    assert load_report(ports["store"]) == {}


def test_save_report_stamps_and_renders(ports):
    save_report(ports["store"], {"header": []}, ports["report"], NOW)
    assert ports["store"].read_json("qa/qa.json") == {"header": [], "updated": "2026-11-22T10:00:00"}
    assert ports["store"].files["qa/qa.md"].startswith("# Contrôles qualité")


def test_flat_comparison_writes_the_ratio_and_reports_moved_dust(ports):
    frames = sept06_frames() + series_frames("flat/SESSION_sept10", FLAT, 1.13, "2021-09-10T13:00:00", 3,
                                             step_s=2, filter="H")
    inv = inventory_of(frames)
    a, b = sorted((s for s in inv.calibration_sets[FLAT] if s.filter == "H"), key=lambda s: s.start)
    yy, xx = np.mgrid[0:64, 0:96]
    ring = 1 - 0.05 * ((np.hypot(yy - 30, xx - 40) > 2) & (np.hypot(yy - 30, xx - 40) < 5))
    ports["pixels"].flats = {a.id: np.ones((64, 96), np.float32), b.id: ring.astype(np.float32)}
    said = []
    items, tables = raw_checks(inv, inv.config, store=ports["store"], pixels=ports["pixels"], progress=said.append)
    rel = f"qa/flats/ratio_{a.id}_vs_{b.id}.fits"
    assert rel in ports["store"].images and ports["store"].images[rel].dtype == np.float32
    (row,) = tables["flat_pairs"]
    assert row["verdict"].startswith("poussières déplacées")
    (p4,) = [i for i in items if i.check == "P4"]
    assert p4.files == [str(ports["store"].root / rel)]
    assert said == ["  contrôles : comparaison des flats H (2 séries) …"]
    # Darks of flat matched to each series give the dark level subtracted from the flats.
    assert [c[3] for c in ports["pixels"].flat_calls] == [516.0, 516.0]


def test_flat_comparison_can_be_disabled(ports):
    frames = sept06_frames() + series_frames("flat/SESSION_sept10", FLAT, 1.13, "2021-09-10T13:00:00", 3,
                                             step_s=2, filter="H")
    inv = inventory_of(frames, make_settings(qa={"flat_compare": False}))
    _, tables = raw_checks(inv, inv.config, store=ports["store"], pixels=ports["pixels"])
    assert "flat_pairs" not in tables and ports["pixels"].flat_calls == []


def test_calibrated_checks_read_frames_and_ampglow_mask():
    rng = np.random.default_rng(0)
    img = rng.normal(50, 5, (160, 240)).astype(np.float32)
    mask = np.zeros_like(img)
    mask[:24, 200:] = 1
    img[mask > 0] -= 20
    pixels = FakePixelStats(images={"/w/cal/a.fit": img, "/w/cal/b.fit": img - 60, "/w/m/ampglow.fits": mask})
    plan = {"groups": {"g1": {"label": "s1", "session": "2021-09-06", "filter": "H", "dark": {"id": "d"},
                              "ampglow": "/w/m/ampglow.fits"},
                       "g2": {"label": "s1", "session": "2021-09-06", "filter": "S", "dark": {"id": "d"},
                              "ampglow": "/w/m/absent.fits"}}}
    results = {"g1": {"outputs": {"light/a.fits": "/w/cal/a.fit"}},
               "g2": {"outputs": {"light/b.fits": "/w/cal/b.fit"}}}
    items, rows = calibrated_checks(plan, results, {"qa": {"calibrated_background_min_adu": 0.0}}, pixels=pixels)
    assert [r["group"] for r in rows] == ["g1", "g2"]
    assert rows[0]["ampglow_zone_resid_adu"][0] == pytest.approx(-20, abs=3)
    assert rows[1]["ampglow_zone_resid_adu"] == []                   # missing mask: no zone
    assert [(i.check, i.severity, i.data.get("group")) for i in items] == [("C3", "error", "g2"),
                                                                         ("C1", "info", "g1")]


def test_record_calibrated_checks_replaces_only_the_recalibrated_groups(ports):
    store, renderer = ports["store"], ports["report"]
    store.write_json("qa/qa.json", {
        "header": [{"check": "H9"}],
        "calibrated": [{"check": "C3", "data": {"group": "g1"}}, {"check": "C3", "data": {"group": "g2"}}],
        "tables": {"calibrated": [{"group": "g1", "n": 1}, {"group": "g2", "n": 1}]}})
    new = QAItem("C1", "info", "x", data={"group": "g1"})
    rep = record_calibrated_checks(store, renderer, [new], [{"group": "g1", "n": 3}], ["g1"], NOW)
    assert [d["data"]["group"] for d in rep["calibrated"]] == ["g2", "g1"]
    assert rep["calibrated"][1]["check"] == "C1"
    assert rep["tables"]["calibrated"] == [{"group": "g1", "n": 3}, {"group": "g2", "n": 1}]
    assert rep["header"] == [{"check": "H9"}] and store.read_json("qa/qa.json") == rep
