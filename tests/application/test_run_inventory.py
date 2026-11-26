"""Use case « inventaire » with in-memory fakes (no file, no FITS)."""

from pathlib import Path

import pytest

from builders import make_settings, sept06_frames
from fakes import FakePixelReader, FakeProvenance, InMemoryArtifactStore, InMemoryFrameSource, RecordingInventoryReport
from varstar.application.inventory import load_inventory, run_inventory

ROOT = Path("/raw")


@pytest.fixture
def ports():
    return {"store": InMemoryArtifactStore(), "provenance": FakeProvenance(),
            "source": InMemoryFrameSource(sept06_frames()), "pixels": FakePixelReader(4),
            "report": RecordingInventoryReport()}


def test_load_inventory_measures_the_adu_scale_once_per_sensor():
    pixels = FakePixelReader(4)
    inv = load_inventory(InMemoryFrameSource(sept06_frames()), pixels, ROOT, make_settings())
    assert len(pixels.read) == 1 and pixels.read[0].startswith("light/H/")
    assert inv.sensors[0].adu_scale == 4
    assert {g.level for g in inv.light_groups} == {"A"}


def test_load_inventory_survives_unreadable_pixels():
    first_light = sorted(f.rel_path for f in sept06_frames() if f.rel_path.startswith("light/"))[0]
    inv = load_inventory(InMemoryFrameSource(sept06_frames()), FakePixelReader(failing=[first_light]), ROOT,
                         make_settings())
    assert inv.sensors[0].adu_scale is None
    assert any(a.code == "adu_scale_failed" and a.files == [first_light] for a in inv.anomalies)


def test_load_inventory_without_pixel_reading():
    pixels = FakePixelReader(4)
    inv = load_inventory(InMemoryFrameSource(sept06_frames()), pixels, ROOT,
                         make_settings(inventory={"read_pixels_for_units": False}))
    assert pixels.read == [] and inv.sensors[0].adu_scale is None


def test_run_inventory_publishes_everything(ports):
    cfg = make_settings()
    res = run_inventory(ROOT, cfg, **ports)
    store, prov = ports["store"], ports["provenance"]
    assert not res.skipped and res.inventory is not None
    assert set(store.files) == {"inventory.json", "inventory.md", "config.used.yaml"}
    assert store.read_json("inventory.json") == res.data == {
        "levels": {"2021-09-06/H": "A", "2021-09-06/S": "A"}, "anomalies": ["offset_missing"]}
    assert ports["report"].qa_items is not None                     # header checks go into inventory.md
    assert prov.manifests["inventory"]["n_files"] == 22
    assert prov.records["inventory"]["outputs"] == ["inventory.json", "inventory.md"]


def test_run_inventory_is_skipped_when_nothing_changed(ports):
    cfg = make_settings()
    first = run_inventory(ROOT, cfg, **ports)
    ports["store"].writes.clear()
    again = run_inventory(ROOT, cfg, **ports)
    assert again.skipped and again.inventory is None and again.data == first.data
    assert ports["store"].writes == [] and ports["source"].scans == 1


@pytest.mark.parametrize("change", ["files", "code", "config", "force"])
def test_run_inventory_reruns_when_something_changed(ports, change):
    cfg = make_settings()
    run_inventory(ROOT, cfg, **ports)
    force = False
    if change == "files":
        ports["provenance"].files_value = "files-v2"
    elif change == "code":
        ports["provenance"].code_value = "code-v2"
    elif change == "config":
        cfg = make_settings(matching={"temp_tolerance_c": 3.0})
    else:
        force = True
    again = run_inventory(ROOT, cfg, force=force, **ports)
    assert not again.skipped and ports["source"].scans == 2


def test_run_inventory_excludes_the_workdir(ports):
    inside = sept06_frames()[0]
    inside.path = ports["store"].root / "calibrated" / "x.fits"
    ports["source"].frames.append(inside)
    res = run_inventory(ROOT, make_settings(), **ports)
    assert ports["provenance"].manifests["inventory"]["n_files"] == 22
    assert sum(g.n for g in res.inventory.light_groups) == 6
