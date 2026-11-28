"""Contracts of the ports: the same tests run on the in-memory fake and on the real adapter.

If a fake drifts from the real behaviour, these tests fail, so the use-case tests (which use the
fakes) stay meaningful.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from astropy.io import fits

from builders import make_frame, make_frame_stats, make_settings
from fakes import (FakeCalibrator, FakeMasterBuilder, FakePixelReader, FakePixelStats, FakeProvenance,
                   InMemoryArtifactStore, InMemoryFrameSource, InMemoryMasterRepository)
from fitsgen import noise, series, write_frame
from varstar.adapters.discovery import FilesystemFrameSource
from varstar.adapters.fits_io import FitsPixelReader, read_frame
from varstar.adapters.images import NORM_SCALE
from varstar.adapters.library import DiskMasterRepository
from varstar.adapters.pixel_stats import PixelStatsCache
from varstar.adapters.siril import Siril
from varstar.adapters.siril_calibrator import SirilCalibrator
from varstar.adapters.siril_masters import SirilMasterBuilder
from varstar.adapters.provenance import ProvenanceStore
from varstar.adapters.workspace import Workspace
from varstar.domain.calibration import CalibrationSensor, MasterRecord, choose_rejection
from varstar.domain.headers import folder_label_from_path
from varstar.domain.plan import DarkRef, GroupPlan
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


# --------------------------------------------------------------------------- FrameSource.file_stat


def test_frame_source_file_stat(frame_source):
    source, root = frame_source
    path = root / FILES[0][0]
    st = source.file_stat(path)
    assert st == source.file_stat(path) and st.resolved.endswith(FILES[0][0]) and st.size > 0


def test_artifact_store_writes_masks(store):
    store.write_mask("masks/bpm_x.fits", np.array([[0, 1], [4, 0]]), {"VSDARK": "d1", "VSBIT1": "hot"})
    store.write_mask("masks/ampglow_x.fits", np.zeros((2, 2)))
    assert store.exists("masks/bpm_x.fits") and store.exists("masks/ampglow_x.fits")


# --------------------------------------------------------------------------- MasterRepository

MASTER_META = {"id": "abc", "kind": "dark", "label": "DARK_g120", "rejection": {"name": "sigma clipping"},
               "siril_version": "1.4.5", "created": "2026-11-26T09:00:00", "n_frames": 3, "exptime": 300.0,
               "median_adu": 500.0}


@pytest.fixture(params=["fake", "real"])
def master_repo(request, tmp_path):
    image = np.full((4, 6), 500.0, np.float32)
    if request.param == "fake":
        repo = InMemoryMasterRepository(tmp_path / "lib")
        repo.add(MasterRecord("abc", "dark", tmp_path / "lib" / "darks" / "DARK_g120_abc", dict(MASTER_META)), image)
        return repo
    repo = DiskMasterRepository(tmp_path / "lib")
    d = repo.new_dir("dark", "DARK_g120", "abc")
    repo.publish(d, image / NORM_SCALE, fits.Header(), dict(MASTER_META), [make_frame(DARK, rel_path="d.fits")])
    return repo


def test_master_repository(master_repo):
    assert master_repo.find("dark", "unknown") is None and master_repo.find("flat", "abc") is None
    rec = master_repo.find("dark", "abc")
    assert (rec.id, rec.kind, rec.meta["median_adu"]) == ("abc", "dark", 500.0)
    adu = master_repo.load_adu(rec)
    assert adu.dtype == np.float32 and np.allclose(adu, 500.0, atol=1e-3)
    assert master_repo.load_unstable(rec) is None
    assert isinstance(master_repo.root, Path)


# --------------------------------------------------------------------------- MasterBuilder, Calibrator (Siril)


def _raw_darks(tmp_path, n=3, level=500.0, exptime=300.0, kind="DARK"):
    paths = series(tmp_path / "raw", f"dark/{exptime:g}s", kind, exptime, "2021-09-07T02:00:00", n, step_s=310,
                   prefix="", shape=(64, 96), make=lambda i: noise(level, 12, 70 + i, shape=(64, 96)))
    return [read_frame(p, tmp_path / "raw") for p in paths]


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.siril)])
def master_builder(request, tmp_path):
    if request.param == "fake":
        repo = InMemoryMasterRepository(tmp_path / "lib")
        return FakeMasterBuilder(repo), repo
    repo = DiskMasterRepository(tmp_path / "lib")
    siril = Siril.from_config(make_settings(), tmp_path / "work")
    return SirilMasterBuilder(siril, repo, tmp_path / "work" / "siril", make_settings()), repo


def test_master_builder_dark_then_flat(master_builder, tmp_path):
    builder, repo = master_builder
    cfg = make_settings()
    darks = _raw_darks(tmp_path)
    rej = choose_rejection(len(darks), cfg.masters)
    dark = builder.build_dark(darks, "dark", rej, "d1", "DARK_g120_ox_b2_300s_T-12", tmp_path / "raw")
    assert (dark.id, dark.kind) == ("d1", "dark") and repo.find("dark", "d1") is not None
    for key in ("label", "n_frames", "exptime", "median_adu", "sources", "frame_stats", "sigma_single_frame_adu"):
        assert key in dark.meta, key
    assert dark.meta["n_frames"] == 3 and dark.meta["rejection"]["name"] == rej.name
    assert np.median(repo.load_adu(dark)) == pytest.approx(dark.meta["median_adu"], abs=2)
    assert abs(dark.meta["median_adu"] - 500) < 5
    assert isinstance(builder.version(), str) and isinstance(builder.status(), dict)

    darkflats = _raw_darks(tmp_path, exptime=1.13, kind="DARKFLAT")
    df = builder.build_dark(darkflats, "darkflat", rej, "df1", "DARKFLAT_g120", tmp_path / "raw")
    flats_paths = series(tmp_path / "raw", "flat", "FLAT", 1.13, "2021-09-07T13:00:00", 3, step_s=2, filt="H",
                         shape=(64, 96), make=lambda i: noise(30000, 100, 80 + i, shape=(64, 96)))
    flats = [read_frame(p, tmp_path / "raw") for p in flats_paths]
    flat = builder.build_flat(flats, df, rej, "f1", "FLAT_H", tmp_path / "raw")
    assert flat.kind == "flat" and repo.find("flat", "f1") is not None and flat.meta["darkflat_id"] == "df1"
    for key in ("level_min_frac", "level_max_frac", "level_warnings", "filter"):
        assert key in flat.meta, key


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.siril)])
def calibrator(request, tmp_path):
    if request.param == "fake":
        store = InMemoryArtifactStore(tmp_path / "work")
        return FakeCalibrator(store), (lambda p: store.exists(str(Path(p).relative_to(store.root))))
    siril = Siril.from_config(make_settings(), tmp_path / "work")
    return SirilCalibrator(siril, tmp_path / "work"), (lambda p: Path(p).exists())


def test_calibrator_writes_one_output_per_light(calibrator, tmp_path):
    calib, exists = calibrator
    lights = series(tmp_path / "raw", "light/H/SESSION_s1", "LIGHT", 300, "2021-09-06T23:00:00", 2, step_s=600,
                    filt="H", shape=(64, 96), make=lambda i: noise(800, 12, 90 + i, shape=(64, 96)))
    dark = tmp_path / "lib" / "dark.fits"
    dark.parent.mkdir(parents=True)
    hdu = fits.PrimaryHDU(np.full((64, 96), 500 / NORM_SCALE, np.float32))
    hdu.header["VSNORM"] = NORM_SCALE
    hdu.writeto(dark)
    group = GroupPlan("2021-09-06_H_300s", "2021-09-06", "s1", "H", 300.0, "B", [], [str(p) for p in lights],
                      DarkRef("d1", str(dark), "direct", 3, 300.0))
    outputs = {str(p): tmp_path / "work" / "calibrated" / f"{i}.fits" for i, p in enumerate(lights)}
    calib.calibrate(group, outputs, CalibrationSensor(4, 0.25))
    assert all(exists(p) for p in outputs.values())


# --------------------------------------------------------------------------- StarCatalog


CATALOG_ROWS = [("1", 60.44, 36.11, 9.5), ("2", 60.50, 36.10, 8.0), ("3", 60.44, 36.60, 10.0),
                ("4", 60.45, 36.12, 14.5), ("5", 61.40, 36.11, 7.0)]


@pytest.fixture(params=["fake", "csv", "gaia-cache"])
def star_catalog(request, tmp_path):
    from fakes import FakeStarCatalog
    from varstar.adapters.star_catalog import CsvStarCatalog, GaiaTapCatalog
    from varstar.domain.astrometry import CatalogStars
    text = "source_id,ra,dec,phot_g_mean_mag\n" + "".join(f"{i},{r},{d},{m}\n" for i, r, d, m in CATALOG_ROWS)
    if request.param == "fake":
        return FakeStarCatalog(CatalogStars(np.array([r[1] for r in CATALOG_ROWS]), np.array([r[2] for r in CATALOG_ROWS]),
                                            np.array([r[3] for r in CATALOG_ROWS])))
    if request.param == "csv":
        p = tmp_path / "cat.csv"
        p.write_text(text)
        return CsvStarCatalog(p)
    table = {"source_id": np.array([int(r[0]) for r in CATALOG_ROWS]), "ra": np.array([r[1] for r in CATALOG_ROWS]),
             "dec": np.array([r[2] for r in CATALOG_ROWS]), "phot_g_mean_mag": np.array([r[3] for r in CATALOG_ROWS])}
    return GaiaTapCatalog(tmp_path / "catalogs", ["https://tap.example/tap"], fetch=lambda url, q: (table, False))


def test_star_catalog_contract(star_catalog):
    st = star_catalog.cone(60.44, 36.11, 0.3, 13.0)
    # In the cone, brighter than the limit, brightest first.
    assert list(st.mag) == [8.0, 9.5]
    assert np.all(np.diff(st.mag) >= 0) and st.source
    assert star_catalog.describe()
    assert star_catalog.cone(60.44, 36.11, 0.3, 9.0).n == 1


# --------------------------------------------------------------------------- PlateSolver


@pytest.fixture(params=["fake", "real"])
def plate_solver(request, tmp_path):
    from builders import make_solution, synthetic_sky
    from fakes import FakePlateSolver
    from varstar.adapters.plate_solver import InternalPlateSolver
    from varstar.domain.astrometry import SolveHint, SolveJob
    shape = (300, 450)
    truth = make_solution(60.44, 36.11, scale_arcsec=3.4877, width=shape[1], height=shape[0])
    img, cat = synthetic_sky(truth, shape, n=700)
    p = tmp_path / "img.fits"
    fits.writeto(p, img)
    hint = SolveHint(60.44, 36.11, 3.4877, shape[1], shape[0])
    jobs = [SolveJob(str(p), hint, "2021-09-06T23:00:00"), SolveJob(str(tmp_path / "nohint.fits"), None, None,
                                                                    "RA/DEC absents")]
    solver = FakePlateSolver() if request.param == "fake" else InternalPlateSolver(1)
    return solver, jobs, cat, truth


def test_plate_solver_contract(plate_solver):
    from varstar.domain.astrometry import SolveSettings
    solver, jobs, cat, truth = plate_solver
    out = solver.solve([jobs], cat, SolveSettings(sip_order=0, min_matches=15))
    assert set(out) == {j.path for j in jobs}
    ok = out[jobs[0].path]
    assert ok.ok and ok.rms_arcsec is not None and ok.n_matched > 0
    x, y = ok.solution.world_to_pixel(*truth.crval)
    assert abs(x - 224.5) < 0.5 and abs(y - 149.5) < 0.5
    assert abs(ok.solution.scale_arcsec - 3.4877) < 0.01
    no = out[jobs[1].path]
    assert not no.ok and no.reason == "RA/DEC absents" and no.solution is None
    assert solver.name()
