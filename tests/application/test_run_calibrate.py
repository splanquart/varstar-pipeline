"""Use case « calibrate » with in-memory fakes (no file, no Siril)."""

import numpy as np
import pytest

from fakes import (FakeCalibrator, FakePixelStats, FakeProvenance, InMemoryArtifactStore, InMemoryFrameSource,
                   RecordingQAReport)
from varstar.application.calibrate import calibrate_group, run_calibrate
from varstar.domain.calibration import CalibrationSensor
from varstar.domain.plan import DarkRef, GroupPlan, MastersPlan

SENSOR = CalibrationSensor(4, 0.25)


def group(key="2021-09-06_H_300s", label="s1", filt="H", level="A", dark=True):
    return GroupPlan(key, "2021-09-06", label, filt, 300.0, level, [],
                     [f"/raw/light/{filt}/a {i}.fits" for i in range(2)],
                     DarkRef("d1", "/lib/d1/master.fits", "direct", 8, 300.0) if dark else None)


@pytest.fixture
def env():
    store = InMemoryArtifactStore()
    return {"store": store, "calibrator": FakeCalibrator(store), "source": InMemoryFrameSource([])}


def test_outputs_are_named_from_the_raw_files(env):
    res = calibrate_group(group(), sensor=SENSOR, **env)
    assert res["skipped"] is False and res["level"] == "A"
    assert res["outputs"] == {"/raw/light/H/a 0.fits": "/work/calibrated/2021-09-06_s1/H/a_0.fits",
                              "/raw/light/H/a 1.fits": "/work/calibrated/2021-09-06_s1/H/a_1.fits"}
    assert env["store"].read_json("manifests/calibrate/2021-09-06_H_300s.json")["fingerprint"] == res["fingerprint"]


def test_unchanged_group_is_skipped(env):
    first = calibrate_group(group(), sensor=SENSOR, **env)
    again = calibrate_group(group(), sensor=SENSOR, **env)
    assert again["skipped"] and again["outputs"] == first["outputs"] and env["calibrator"].calls == [first["key"]]


@pytest.mark.parametrize("change", ["force", "raw touched", "output missing", "level"])
def test_group_is_recalibrated_when_something_changed(env, change):
    g = group()
    calibrate_group(g, sensor=SENSOR, **env)
    force = change == "force"
    if change == "raw touched":
        env["source"].mtimes[g.files[0]] = 5
    elif change == "output missing":
        del env["store"].files["calibrated/2021-09-06_s1/H/a_0.fits"]
    elif change == "level":
        g.level = "B"
    assert calibrate_group(g, sensor=SENSOR, force=force, **env)["skipped"] is False
    assert len(env["calibrator"].calls) == 2


def test_run_calibrate_isolates_failures_and_runs_the_checks(env):
    g_ok, g_bad = group(), group("2021-09-06_S_300s", filt="S")
    plan = MastersPlan("t", "/lib", "1.4.5", {g_ok.key: g_ok, g_bad.key: g_bad})
    env["calibrator"].failing = {g_bad.key}
    images = {out: np.full((16, 16), -5.0) for out in
              ("/work/calibrated/2021-09-06_s1/H/a_0.fits", "/work/calibrated/2021-09-06_s1/H/a_1.fits")}
    said, prov = [], FakeProvenance()
    res = run_calibrate(plan, [g_ok.key, g_bad.key, "absent"], {"qa": {"calibrated_background_min_adu": 0.0}},
                        SENSOR, pixels=FakePixelStats(images=images), qa_report=RecordingQAReport(), provenance=prov,
                        tool_status={"active": True}, progress=said.append, **env)
    assert list(res["results"]) == [g_ok.key]
    assert res["errors"] == {g_bad.key: "Siril a échoué",
                             "absent": "groupe absent du plan des masters (relancer l'étape masters)"}
    assert said == ["  calibration s1 / H (2 images) …", "  calibration s1 / S (2 images) …"]
    assert [d["check"] for d in res["qa"]] == ["C3"]                         # negative background
    assert env["store"].read_json("qa/qa.json")["calibrated"] == res["qa"]
    record = prov.records["calibrate"]
    assert set(record["images"]) == set(g_ok.files) and record["groups"][g_ok.key]["dark"]["id"] == "d1"
    assert record["errors"] == res["errors"] and record["siril_python_init_guard"] == {"active": True}


def test_run_calibrate_without_any_result(env):
    plan = MastersPlan("t", "/lib", "1.4.5", {})
    prov = FakeProvenance()
    res = run_calibrate(plan, ["x"], {}, SENSOR, pixels=FakePixelStats(), qa_report=RecordingQAReport(),
                        provenance=prov, **env)
    assert res["results"] == {} and prov.records["calibrate"]["groups"] == {} and res["qa"] == []
