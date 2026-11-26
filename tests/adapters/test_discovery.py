"""Discovery of FITS files on disk and the FilesystemFrameSource adapter."""

from fitsgen import write_frame
from varstar.adapters.discovery import FilesystemFrameSource, discover
from varstar.domain.model import LIGHT


def test_discovery_ignores(tmp_path, cfg):
    write_frame(tmp_path / "light/a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "removed/b.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "light/S/SESSION_x/removed/c.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    write_frame(tmp_path / "output/master/d.fit", "DARK", 300, "2021-09-06T23:00:00")
    write_frame(tmp_path / "proj.pxiproject/e.fits", "DARK", 300, "2021-09-06T23:00:00")
    write_frame(tmp_path / "light/._a.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    (tmp_path / "light/img.xisf").write_text("x")
    (tmp_path / "bias/SESSION_1").mkdir(parents=True)
    res = discover(tmp_path, cfg)
    assert [p.name for p in res.fits_files] == ["a.fits"]
    assert "bias/SESSION_1" in res.empty_dirs
    assert res.other_extensions[".xisf"] == 1
    cfg["inventory"]["include_removed"] = True
    assert sorted(p.name for p in discover(tmp_path, cfg).fits_files) == ["a.fits", "b.fits", "c.fits"]


def test_frame_source_scan(tmp_path, cfg):
    root = tmp_path / "raw"
    write_frame(root / "light/H/SESSION_s1/a b.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    (root / "light/H/broken.fits").write_bytes(b"not a fits file")
    write_frame(root / "work/x.fits", "LIGHT", 300, "2021-09-06T23:00:00", filt="H")
    (root / "notes.txt").write_text("x")
    source = FilesystemFrameSource()
    files = source.files(root, cfg, exclude=[root / "work"])
    assert sorted(p.name for p in files) == ["a b.fits", "broken.fits"]
    scan = source.scan(root, cfg, exclude=[root / "work"])
    assert scan.root == root.resolve()
    (f,) = scan.frames
    assert (f.imagetyp, f.rel_path, f.folder_label) == (LIGHT, "light/H/SESSION_s1/a b.fits", "s1")
    assert len(scan.unreadable) == 1 and scan.unreadable[0].startswith("light/H/broken.fits (")
    assert scan.other_extensions == {".txt": 1}
