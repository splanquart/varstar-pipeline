"""Contracts of the ports: the same tests run on the in-memory fake and on the real adapter.

If a fake drifts from the real behaviour, these tests fail, so the use-case tests (which use the
fakes) stay meaningful.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from builders import make_frame, make_frame_stats, make_settings
from fakes import FakePixelReader, FakePixelStats, FakeProvenance, InMemoryArtifactStore, InMemoryFrameSource
from fitsgen import noise, write_frame
from varstar.adapters.discovery import FilesystemFrameSource
from varstar.adapters.fits_io import FitsPixelReader
from varstar.adapters.pixel_stats import PixelStatsCache
from varstar.adapters.provenance import ProvenanceStore
from varstar.adapters.workspace import Workspace
from varstar.domain.headers import folder_label_from_path
from varstar.domain.model import DARK, LIGHT

FILES = [  # (relative path, type, filter)
    ("light/H/SESSION_s1/2021-09-06_23-00-00_NGC 1499_H.fits", LIGHT, "H"),
    ("light/H/SESSION_s1/2021-09-06_23-10-00_NGC 1499_H.fits", LIGHT, "H"),
    ("dark/300s/2021-09-07_02-00-00_Dark.fits", DARK, "Dark"),   # N.I.N.A. writes FILTER = Dark
    ("removed/2021-09-06_23-20-00_NGC 1499_H.fits", LIGHT, "H"),       # ignored folder
    ("work/calibrated/x.fits", LIGHT, "H"),                            # excluded (work directory)
]


# --------------------------------------------------------------------------- FrameSource


@pytest.fixture(params=["fake", "real"])
def frame_source(request, tmp_path):
    root = tmp_path / "raw"
    if request.param == "real":
        for i, (rel, kind, filt) in enumerate(FILES):
            write_frame(root / rel, kind, 300, f"2021-09-06T23:{i:02d}:00", filt=filt)
        return FilesystemFrameSource(), root
    frames = []
    for rel, kind, filt in FILES:
        f = make_frame(kind, filter=filt, rel_path=rel, folder_label=folder_label_from_path(rel))
        f.path = root / rel
        frames.append(f)
    return InMemoryFrameSource(frames), root


def test_frame_source_files_skip_ignored_and_excluded(frame_source):
    source, root = frame_source
    files = source.files(root, make_settings(), exclude=[root / "work"])
    assert sorted(str(p.relative_to(root)) for p in files) == sorted(r for r, _, _ in FILES[:3])


def test_frame_source_scan_matches_files(frame_source):
    source, root = frame_source
    cfg = make_settings()
    scan = source.scan(root, cfg, exclude=[root / "work"])
    assert scan.root == root
    assert [f.rel_path for f in scan.frames] == sorted(r for r, _, _ in FILES[:3])
    assert {f.rel_path: (f.imagetyp, f.filter, f.folder_label) for f in scan.frames} == {
        r: (k, flt, folder_label_from_path(r)) for r, k, flt in FILES[:3]}
    assert scan.unreadable == []


def test_frame_source_scans_are_independent(frame_source):
    source, root = frame_source
    cfg = make_settings()
    first = source.scan(root, cfg)
    first.frames[0].filter = "changed"
    assert source.scan(root, cfg).frames[0].filter != "changed"


# --------------------------------------------------------------------------- ArtifactStore


@pytest.fixture(params=["fake", "real"])
def store(request, tmp_path):
    return InMemoryArtifactStore(tmp_path / "work") if request.param == "fake" else Workspace(tmp_path / "work")


def test_artifact_store_round_trips(store):
    assert not store.exists("inventory.json") and store.read_json("inventory.json") is None
    data = {"message": "Généré — niveau B", "n": 3, "items": [1.5, None, True]}
    store.write_json("qa/qa.json", data)
    assert store.exists("qa/qa.json") and store.read_json("qa/qa.json") == data
    store.write_text("inventory.md", "v1")
    store.write_text("inventory.md", "v2")
    assert store.exists("inventory.md")
    store.write_yaml("config.used.yaml", {"masks": {"hot_pixel_sigma": 5.0}})
    assert store.exists("config.used.yaml")
    assert isinstance(store.root, Path)


def test_artifact_store_writes_images_and_tolerates_dates(store):
    store.write_image("qa/flats/ratio.fits", np.ones((4, 6), dtype=np.float64))
    store.write_json("qa/x.json", {"night": dt.date(2021, 9, 6)})
    assert store.read_json("qa/x.json") == {"night": "2021-09-06"}


# --------------------------------------------------------------------------- ProvenanceRecorder


@pytest.fixture(params=["fake", "real"])
def provenance(request, tmp_path):
    return FakeProvenance() if request.param == "fake" else ProvenanceStore(tmp_path / "work")


def test_provenance_manifests_round_trip(provenance):
    assert provenance.read_manifest("inventory") is None
    provenance.write_manifest("inventory", {"fingerprint": {"files": "x"}, "n_files": 2})
    assert provenance.read_manifest("inventory") == {"fingerprint": {"files": "x"}, "n_files": 2}
    provenance.record("inventory", make_settings(), {"n_files": 2})


def test_provenance_fingerprints(provenance, tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    a.write_bytes(b"a")
    b.write_bytes(b"bb")
    assert provenance.files_fingerprint([a]) == provenance.files_fingerprint([a])
    assert provenance.files_fingerprint([a]) != provenance.files_fingerprint([a, b])
    assert provenance.code_fingerprint() == provenance.code_fingerprint()
    cfg = make_settings()
    other_section = make_settings(runtime={"processes": 1})
    other_matching = make_settings(matching={"temp_tolerance_c": 3.0})
    sections = ["inventory", "matching"]
    assert provenance.config_fingerprint(cfg, sections) == provenance.config_fingerprint(other_section, sections)
    assert provenance.config_fingerprint(cfg, sections) != provenance.config_fingerprint(other_matching, sections)


# --------------------------------------------------------------------------- PixelReader


@pytest.fixture(params=["fake", "real"])
def pixel_reader(request, tmp_path):
    path = write_frame(tmp_path / "a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H", scale=4)
    frame = make_frame(LIGHT, rel_path="a.fits")
    frame.path = path
    return (FakePixelReader(4) if request.param == "fake" else FitsPixelReader()), frame


def test_pixel_reader_adu_scale(pixel_reader):
    reader, frame = pixel_reader
    assert reader.adu_scale(frame) == 4


# --------------------------------------------------------------------------- PixelStats

LEVEL, SIGMA = 1000.0, 12.0


@pytest.fixture(params=["fake", "real"])
def pixel_stats(request, tmp_path):
    """Three dark frames around LEVEL ADU (×4 data), and an image of the work directory."""
    frames = []
    for i in range(3):
        f = make_frame(DARK, rel_path=f"dark/{i}.fits")
        f.path = tmp_path / "raw" / f.rel_path
        frames.append(f)
    image = tmp_path / "work" / "calibrated" / "a.fit"
    if request.param == "real":
        for i, f in enumerate(frames):
            write_frame(f.path, "DARK", 300, "2021-09-07T02:00:00", data=noise(LEVEL, SIGMA, i, shape=(64, 96)))
        Workspace(tmp_path / "work").write_image("calibrated/a.fit", np.full((8, 8), 3.5))
        return PixelStatsCache(tmp_path / "work", step=1), frames, image
    stats = {f.rel_path: make_frame_stats(LEVEL, SIGMA) for f in frames}
    return FakePixelStats(stats, images={image: np.full((8, 8), 3.5)}), frames, image


def test_pixel_stats_frame_stats(pixel_stats):
    pixels, frames, _ = pixel_stats
    stats = pixels.frame_stats(frames, 65532.0)
    assert list(stats) == [str(f.path) for f in frames]
    for st in stats.values():
        assert set(st) >= set(make_frame_stats())
        assert st["median"] == pytest.approx(LEVEL, abs=2) and st["robust_sigma"] == pytest.approx(SIGMA, rel=0.15)
        assert st["adu_scale"] == 4 and st["frac_saturated"] == 0


def test_pixel_stats_quick_flat_is_normalised(pixel_stats):
    pixels, frames, _ = pixel_stats
    flat = pixels.quick_flat("FLAT-x", frames, 8, 0.0)
    assert flat.dtype == np.float32 and flat.ndim == 2 and np.median(flat) == pytest.approx(1)


def test_pixel_stats_read_image(pixel_stats, tmp_path):
    pixels, _, image = pixel_stats
    img = pixels.read_image(image)
    assert img.dtype == np.float32 and img.shape == (8, 8) and np.all(img == 3.5)
    with pytest.raises(FileNotFoundError):
        pixels.read_image(tmp_path / "work" / "absent.fits")
