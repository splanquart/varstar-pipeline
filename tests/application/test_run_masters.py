"""Use case « masters » with in-memory fakes (no file, no Siril)."""

import datetime as dt

import numpy as np
import pytest

from builders import inventory_of, make_frame_stats, make_settings, sept06_frames, series_frames, two_nights_frames
from fakes import (FakeMasterBuilder, FakePixelStats, FakeProvenance, FakeReferenceMasters, InMemoryArtifactStore,
                   InMemoryFrameSource, InMemoryMasterRepository, RecordingMastersReport)
from varstar.application.masters import load_plan, run_masters
from varstar.domain.masks import BIT_NAMES
from varstar.domain.model import DARK, LIGHT
from varstar.domain.qa.model import QAItem

NOW = dt.datetime(2026, 11, 26, 9, 0, 0)


def level_by_folder(f):
    return make_frame_stats(541.0 if f.rel_path.startswith("dark/b") else 516.0)


@pytest.fixture
def ports():
    repo = InMemoryMasterRepository()
    return {"source": InMemoryFrameSource([]), "pixels": FakePixelStats(default=level_by_folder),
            "builder": FakeMasterBuilder(repo), "repo": repo, "references": None, "store": InMemoryArtifactStore(),
            "provenance": FakeProvenance(), "report": RecordingMastersReport()}


def run(inv, ports, cfg=None, groups=None):
    return run_masters(inv, inv.light_groups if groups is None else groups, cfg or inv.config, now=NOW, **ports)


def test_level_a_session_gets_dark_flat_and_masks(ports):
    inv = inventory_of(sept06_frames())
    plan = run(inv, ports)
    assert sorted(k for k, _, _ in ports["builder"].built) == ["dark", "darkflat", "darkflat", "flat", "flat"]
    g = plan.groups["2021-09-06_H_300s"]
    assert (g.level, g.dark.mode, g.dark.n, g.flat.n, g.darkflat.exptime) == ("A", "direct", 4, 3, 1.13)
    assert g.bpm == f"/work/masks/bpm_{g.dark.id}_{g.flat.id}_" + g.bpm.rsplit("_", 1)[1]
    assert g.ampglow.startswith(f"/work/masks/ampglow_{g.dark.id}_")
    store = ports["store"]
    bpm_rel = g.bpm.removeprefix("/work/")
    header = store.headers[bpm_rel]
    assert header["VSDARK"] == g.dark.id and header["VSFLAT"] == g.flat.id
    assert sorted(v for k, v in header.items() if k.startswith("VSBIT")) == sorted(BIT_NAMES.values())
    assert store.read_json(bpm_rel[:-5] + ".json")["counts"] is not None
    # Published: plan.json (historical layout), masters.md, provenance.
    assert store.read_json("masters/plan.json") == plan.to_dict() and load_plan(store) == plan
    assert plan.created == "2026-11-26T09:00:00" and plan.siril_version == "1.4.5"
    assert store.files["masters/masters.md"].startswith("# Masters")
    _, qa_items, masks = ports["report"].calls[-1]
    assert qa_items is None and set(masks) == {g.bpm, g.ampglow, plan.groups["2021-09-06_S_300s"].bpm}
    record = ports["provenance"].records["masters"]
    assert set(record["masters"]) == set(plan.masters) and record["library"] == "/lib"
    assert {m["kind"] for m in plan.masters.values()} == {"dark", "darkflat", "flat"}
    dark = plan.masters[g.dark.id]
    assert dark["consistency"][0]["folder"] == "dark/darklight/300s"


def test_masters_are_reused_when_inputs_are_identical(ports):
    inv = inventory_of(sept06_frames())
    first = run(inv, ports)
    ports["builder"].built.clear()
    store = ports["store"]
    store.writes.clear()
    again = run(inv, ports)
    assert ports["builder"].built == [] and again.groups == first.groups
    assert not any(w.startswith("masks/") for w in store.writes)        # masks already there
    # A raw dark touched on disk → new dark id → rebuilt (masks too).
    dark = next(f for f in inv.frames if f.imagetyp == DARK)
    ports["source"].mtimes[str(dark.path)] = 1
    third = run(inv, ports)
    assert [k for k, _, _ in ports["builder"].built] == ["dark"]
    assert third.groups["2021-09-06_H_300s"].dark.id != first.groups["2021-09-06_H_300s"].dark.id


def test_reduced_mask_blocks_are_warned_once_even_when_masks_are_reused(ports):
    # https://github.com/splanquart/varstar-pipeline/issues/4: the fake masters are 64x96 px.
    inv = inventory_of(sept06_frames())
    plan = run(inv, ports)
    notes = [w for w in plan.warnings if "au lieu de" in w]
    assert len(notes) == 2
    assert any("masque d'amp glow : blocs de 16 px au lieu de 64 px" in w for w in notes)
    assert any("masque BPM, niveau local du flat : blocs de 16 px au lieu de 32 px" in w for w in notes)
    again = run(inv, ports)
    assert [w for w in again.warnings if "au lieu de" in w] == notes


def test_no_block_warning_at_the_reference_size(ports):
    ports["builder"].shape = (512, 768)
    plan = run(inventory_of(sept06_frames()), ports)
    assert not [w for w in plan.warnings if "au lieu de" in w]


def test_qa_findings_go_into_masters_md(ports):
    ports["store"].write_json("qa/qa.json", {"header": [QAItem("H9", "info", "peu de darks").to_dict()],
                                             "calibrated": [QAItem("C3", "error", "x").to_dict()]})
    run(inventory_of(sept06_frames()), ports)
    _, qa_items, _ = ports["report"].calls[-1]
    assert [i.check for i in qa_items] == ["H9"]


def test_scaled_dark_is_level_b_with_an_offset_master(ports):
    frames = [f for f in sept06_frames() if not f.rel_path.startswith("dark/darklight")]
    frames += series_frames("dark/darklight/180s", DARK, 180, "2021-09-07T02:00:00", 4, step_s=200)
    plan = run(inventory_of(frames), ports)
    d = plan.groups["2021-09-06_H_300s"].dark
    assert d.mode == "scaled" and d.exptime == 180 and d.offset_adu == 500.0
    offset_master = ports["repo"].find("darkflat", d.offset_master)
    assert offset_master.meta["exptime"] == 1.13
    assert any("mis à l'échelle depuis 180 s" in w and "niveau B" in w for w in plan.warnings)
    assert plan.to_dict()["groups"]["2021-09-06_H_300s"]["dark"]["offset_master"] == d.offset_master


def test_borrowed_darks_of_an_anomalous_night_are_replaced(ports):
    frames = two_nights_frames() + [
        *series_frames("dark/c", DARK, 300, "2021-09-01T01:00:00", 5, step_s=310),
        *series_frames("light/H/SESSION_sept12", LIGHT, 300, "2021-09-12T23:00:00", 2, step_s=600, filter="H"),
    ]
    inv = inventory_of(frames)
    plan = run(inv, ports, groups=[g for g in inv.light_groups if g.label == "sept12"])
    (g,) = plan.groups.values()
    assert "offset anormal" in g.reasons[-1] and any("offset anormal" in w for w in plan.warnings)
    assert all("dark/a" in s for s in ports["repo"].find("dark", g.dark.id).meta["sources"])
    assert set(ports["pixels"].measured) >= {f.rel_path for f in inv.frames if f.rel_path.startswith("dark/")}


def test_level_c_without_flat_still_gets_a_bad_pixel_map(ports):
    inv = inventory_of([f for f in sept06_frames() if not f.rel_path.startswith("flat/")])
    plan = run(inv, ports)
    g = plan.groups["2021-09-06_H_300s"]
    assert g.flat is None and g.darkflat is None and "_noflat_" in g.bpm
    assert [k for k, _, _ in ports["builder"].built] == ["dark"]


def test_no_master_no_mask(ports):
    inv = inventory_of([f for f in sept06_frames() if f.imagetyp == LIGHT])
    g = run(inv, ports).groups["2021-09-06_H_300s"]
    assert (g.dark, g.flat, g.bpm, g.ampglow) == (None, None, None, None)


def test_amp_glow_detection_can_be_disabled(ports):
    cfg = make_settings(masks={"amp_glow_detect": False})
    inv = inventory_of(sept06_frames(), cfg)
    assert run(inv, ports, cfg).groups["2021-09-06_H_300s"].ampglow is None


def test_flats_out_of_range_are_warned(ports):
    ports["builder"].flat_level_frac = 0.95
    plan = run(inventory_of(sept06_frames()), ports)
    assert sum("hors de la plage 15–85 %" in w for w in plan.warnings) == 2


def test_crosscheck_with_reference_masters(ports):
    ports["references"] = FakeReferenceMasters({"dark": np.full((64, 96), 498, np.float32),
                                                "flat": ValueError("XISF compressé"), "darkflat": None})
    plan = run(inventory_of(sept06_frames()), ports)
    by_kind = {}
    for m in plan.masters.values():
        by_kind.setdefault(m["kind"], m)
    assert by_kind["dark"]["pixinsight"]["median_diff_adu"] == pytest.approx(2.0)
    assert by_kind["flat"]["pixinsight"]["error"] == "XISF compressé"
    assert "pixinsight" not in by_kind["darkflat"]
