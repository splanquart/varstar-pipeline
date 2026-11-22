"""Contracts of the ports: the same tests run on the in-memory fake and on the real adapter.

If a fake drifts from the real behaviour, these tests fail, so the use-case tests (which use the
fakes) stay meaningful.
"""

from pathlib import Path

import pytest

from builders import make_frame, make_settings
from fakes import FakeProvenance, InMemoryArtifactStore, InMemoryFrameSource
from fitsgen import write_frame
from varstar.adapters.discovery import FilesystemFrameSource
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
