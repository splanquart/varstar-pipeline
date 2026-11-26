"""Use case « run » with in-memory fakes: order of the steps, ``--from``, errors."""

from pathlib import Path

import numpy as np
import pytest

from builders import make_frame_stats, make_settings, sept06_frames
from fakes import (FakeCalibrator, FakeMasterBuilder, FakePixelReader, FakePixelStats, FakeProvenance,
                   InMemoryArtifactStore, InMemoryFrameSource, InMemoryMasterRepository, RecordingInventoryReport,
                   RecordingMastersReport, RecordingQAReport, RecordingReporter)
from varstar.application.pipeline import IMPLEMENTED, STEPS, PipelineServices, check_step, run_pipeline
from varstar.domain.calibration import PipelineError
from varstar.domain.model import LIGHT

ROOT = Path("/raw")


@pytest.fixture
def svc():
    store, repo = InMemoryArtifactStore(), InMemoryMasterRepository()
    tools = {"calls": 0}

    def calibration_tools():
        tools["calls"] += 1
        return FakeMasterBuilder(repo), FakeCalibrator(store)

    logs = []
    s = PipelineServices(
        store=store, provenance=FakeProvenance(), source=InMemoryFrameSource(sept06_frames()),
        pixel_reader=FakePixelReader(4), pixel_stats=FakePixelStats(default=lambda f: make_frame_stats(516.0),
                                   default_image=np.full((16, 16), 40.0)),
        inventory_report=RecordingInventoryReport(), qa_report=RecordingQAReport(),
        masters_report=RecordingMastersReport(), repo=repo, references=None, calibration_tools=calibration_tools,
        reporter=RecordingReporter(), start_log=logs.append)
    s.logs, s.tools = logs, tools
    return s


def test_full_run(svc):
    summary = run_pipeline(ROOT, make_settings(), svc)
    assert summary["steps"] == ["inventory", "check", "masters", "calibrate"]
    assert summary["groups"] == ["2021-09-06_H_300s", "2021-09-06_S_300s"]
    assert svc.logs == ["inventory", "masters", "calibrate"]
    assert summary["not_implemented"] == [s for s in STEPS if s not in IMPLEMENTED]
    assert set(summary["calibrate"]["results"]) == set(summary["groups"])
    assert summary["plan"] == svc.store.read_json("masters/plan.json")
    assert {"inventory.json", "qa/qa.json", "masters/plan.json"} <= set(svc.store.files)
    assert svc.reporter.lines[0] == "  masters pour sept06 / H …"
    assert svc.reporter.lines[-1] == "  calibration sept06 / S (3 images) …"
    assert svc.source.scans == 1          # the inventory built by the first step is reused


def test_resume_from_calibrate_uses_the_saved_plan(svc, cfg):
    run_pipeline(ROOT, cfg, svc)
    svc.logs.clear()
    again = run_pipeline(ROOT, cfg, svc, "calibrate")
    assert again["steps"] == ["calibrate"] and svc.logs == ["calibrate"]
    assert all(r["skipped"] for r in again["calibrate"]["results"].values())


def test_resume_without_plan_is_an_error(svc, cfg):
    with pytest.raises(PipelineError, match="plan des masters absent"):
        run_pipeline(ROOT, cfg, svc, "calibrate")


def test_selection(svc, cfg):
    summary = run_pipeline(ROOT, cfg, svc, sessions=["sept06"], filters=["S"])
    assert summary["groups"] == ["2021-09-06_S_300s"]
    with pytest.raises(PipelineError, match="inconnue"):
        run_pipeline(ROOT, cfg, svc, sessions=["sept99"])
    with pytest.raises(PipelineError, match="Aucun groupe"):
        run_pipeline(ROOT, cfg, svc, filters=["O"])


def test_unknown_step(svc, cfg):
    with pytest.raises(PipelineError, match="Étape inconnue : photo"):
        run_pipeline(ROOT, cfg, svc, "photo")
    assert check_step("masters") == 2


def test_later_steps_are_listed_as_not_implemented(svc, cfg):
    run_pipeline(ROOT, cfg, svc)
    summary = run_pipeline(ROOT, cfg, svc, "photometry")
    assert summary["steps"] == [] and summary["not_implemented"][0] == "photometry"


def test_no_light_no_group(svc, cfg):
    svc.source.frames = [f for f in sept06_frames() if f.imagetyp != LIGHT]
    with pytest.raises(PipelineError, match="Aucun groupe"):
        run_pipeline(ROOT, cfg, svc)
    assert svc.tools["calls"] == 0         # Siril is never started for nothing
