"""Use case « astrometry » with fakes: jobs from the headers, cache, segments, A* checks, errors."""

from pathlib import Path

import numpy as np
import pytest

from builders import make_frame_stats, make_settings, sept06_frames
from fakes import (FakeCalibrator, FakeMasterBuilder, FakePixelReader, FakePixelStats, FakePlateSolver,
                   FakeProvenance, FakeStarCatalog, InMemoryArtifactStore, InMemoryFrameSource,
                   InMemoryMasterRepository, RecordingAstrometryReport, RecordingInventoryReport,
                   RecordingMastersReport, RecordingQAReport, RecordingReporter)
from varstar.application.astrometry import build_job, load_solutions, run_astrometry
from varstar.application.inventory import load_inventory
from varstar.application.masters import load_plan
from varstar.application.pipeline import PipelineServices, run_pipeline
from varstar.domain.calibration import PipelineError

ROOT = Path("/raw")


class Run:
    """A pipeline run up to « calibrate » with fakes, then the astrometry use case on its outputs."""

    def __init__(self, frames=None, cfg=None, tools=True):
        self.cfg = cfg or make_settings()
        self.store, repo = InMemoryArtifactStore(), InMemoryMasterRepository()
        self.source = InMemoryFrameSource(frames or sept06_frames())
        self.prov = FakeProvenance()
        svc = PipelineServices(
            store=self.store, provenance=self.prov, source=self.source, pixel_reader=FakePixelReader(4),
            pixel_stats=FakePixelStats(default=lambda f: make_frame_stats(516.0),
                                       default_image=np.full((16, 16), 40.0)),
            inventory_report=RecordingInventoryReport(), qa_report=RecordingQAReport(),
            masters_report=RecordingMastersReport(), repo=repo, references=None,
            calibration_tools=lambda: (FakeMasterBuilder(repo), FakeCalibrator(self.store)),
            reporter=RecordingReporter(), start_log=lambda step: None,
            astrometry_tools=(lambda: (FakePlateSolver(), FakeStarCatalog())) if tools else None,
            astrometry_report=RecordingAstrometryReport())
        self.summary = run_pipeline(ROOT, self.cfg, svc)
        # Start each test from the calibrated images only.
        for rel in [r for r in self.store.files if r.startswith(("astrometry/", "manifests/astrometry/"))]:
            del self.store.files[rel]
        self.plan = load_plan(self.store)
        self.inv = load_inventory(self.source, FakePixelReader(4), ROOT, self.cfg)
        self.keys = self.summary["groups"]
        self.report = RecordingAstrometryReport()
        self.reporter = RecordingReporter()

    def astrometry(self, solver=None, catalog=None, **kw):
        self.solver = solver or FakePlateSolver()
        self.catalog = catalog or FakeStarCatalog()
        return run_astrometry(self.plan, kw.pop("keys", self.keys), self.cfg, self.inv, solver=self.solver,
                              catalog=self.catalog, store=self.store, provenance=self.prov,
                              qa_report=RecordingQAReport(), report=self.report, progress=self.reporter.progress,
                              **kw)


def test_pipeline_without_astrometry_tools_is_an_error():
    with pytest.raises(PipelineError, match="astrométrie non configurée"):
        Run(tools=False)


@pytest.fixture
def ready():
    return Run()


def test_solves_every_calibrated_image(ready):
    out = ready.astrometry()
    assert out["n_images"] == 6 and out["n_solved"] == 6 and out["errors"] == {}
    # One catalogue cone for the whole run, one series per group.
    (ra, dec, radius, mag), = ready.catalog.calls
    assert abs(ra - 60.44) < 1e-3 and abs(dec - 36.11) < 1e-3 and 2.9 < radius < 3.1 and mag == 13.0
    assert len(ready.solver.calls) == 2 and all(len(c) == 3 for c in ready.solver.calls)
    sol = load_solutions(ready.store, "2021-09-06_H_300s")
    row = next(iter(sol["images"].values()))
    assert row["ok"] and row["segment"] == 1 and row["segment_id"] == "2021-09-06/1"
    assert row["header"]["CTYPE1"] == "RA---TAN" and row["calibrated"].endswith(".fits")
    assert "astrometry/astrometry.md" in ready.store.files
    assert ready.prov.records["astrometry"]["solver"] == "faux solveur"
    assert ready.reporter.lines[-1] == "  astrométrie de 6 image(s) (2 groupe(s), 1 étoiles de référence) …"
    qa = ready.store.read_json("qa/qa.json")
    assert qa["astrometry"] == [] and [r["n_solved"] for r in qa["tables"]["astrometry"]] == [3, 3]


def test_second_run_is_skipped_unless_forced_or_changed(ready):
    ready.astrometry()
    again = ready.astrometry()
    assert ready.solver.calls == [] and all(g["skipped"] for g in again["groups"].values())
    assert ready.catalog.calls == []          # nothing to solve: no catalogue query
    ready.astrometry(force=True)
    assert len(ready.solver.calls) == 2


def test_meridian_flip_gives_two_segments_over_both_filters(ready):
    # H lights at 23:04, 23:14, 23:24, S at 23:09, 23:19, 23:29: flip at 23:15 (the filters interleave).
    early = ("_23-04", "_23-09", "_23-14")
    flip = lambda path: 180.6 if any(e in Path(path).name for e in early) else 0.6   # noqa: E731
    out = ready.astrometry(FakePlateSolver(orientation=flip))
    segs = {Path(raw).name[:19]: r["segment"] for g in out["groups"].values() for raw, r in g["images"].items()}
    assert set(segs.values()) == {1, 2}
    assert {name[11:16]: s for name, s in segs.items()} == {"23-04": 1, "23-09": 1, "23-14": 1, "23-19": 2,
                                                            "23-24": 2, "23-29": 2}
    assert [i["check"] for i in out["qa"]] == ["A4"]


def test_failures_and_missing_headers(ready):
    out = ready.astrometry(FakePlateSolver(failing=["_23-04-00"]))
    assert out["n_solved"] == 5
    a1 = [i for i in out["qa"] if i["check"] == "A1"]
    assert len(a1) == 1 and a1[0]["severity"] == "warning"
    # A light without RA/DEC is not solved (no blind solving) and says why.
    frames = sept06_frames()
    for f in frames:
        if f.filter == "S" and f.imagetyp == "LIGHT":
            f.ra = None
    r = Run(frames)
    out = r.astrometry(keys=["2021-09-06_S_300s"])
    assert out["n_solved"] == 0
    row = next(iter(out["groups"]["2021-09-06_S_300s"]["images"].values()))
    assert "RA/DEC absents" in row["reason"]
    assert r.catalog.calls == []


def test_missing_calibration_and_catalogue_error(ready):
    out = ready.astrometry(keys=["2021-09-06_H_300s", "2099-01-01_H_300s"])
    assert "relancer l'étape « calibrate »" in out["errors"]["2099-01-01_H_300s"]
    with pytest.raises(PipelineError, match="hors ligne"):
        ready.astrometry(catalog=FakeStarCatalog(error="hors ligne"), force=True)


def test_build_job_from_the_header():
    cfg = make_settings()
    f = sept06_frames()[0]
    job = build_job("/work/x.fits", f, cfg)
    assert job.hint.width == f.naxis1 and abs(job.hint.scale_arcsec - 3.4855) < 1e-3 and job.time
    f.focallen = None
    assert "FOCALLEN" in build_job("/work/x.fits", f, cfg).problem
    fixed = make_settings(astrometry={"pixel_scale_arcsec": 3.49})
    assert build_job("/work/x.fits", f, fixed).hint.scale_arcsec == 3.49
    assert "introuvable" in build_job("/work/x.fits", None, cfg).problem


def test_a_partial_run_keeps_the_other_groups_findings(ready):
    ready.astrometry(FakePlateSolver(failing=["_23-04-00_H"]))      # both groups, one H image fails
    ready.astrometry(keys=["2021-09-06_S_300s"], force=True)         # S again only
    qa = ready.store.read_json("qa/qa.json")
    assert [i["check"] for i in qa["astrometry"]] == ["A1"]           # the H failure is still reported
    assert [r["group"] for r in qa["tables"]["astrometry"]] == ["2021-09-06_H_300s", "2021-09-06_S_300s"]


def test_flat_borrowing_is_known_from_the_inventory(ready):
    ready.astrometry()
    # sept06 lights use their own flats: no rotation question can arise from them.
    qa = ready.store.read_json("qa/qa.json")
    assert not [i for i in qa["astrometry"] if i["check"] == "A5"]
