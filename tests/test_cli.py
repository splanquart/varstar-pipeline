import json

from typer.testing import CliRunner

from varstar.cli import app

runner = CliRunner()


def test_inventory_command_writes_outputs_and_caches(complete_session, tmp_path):
    wd = tmp_path / "work"
    raw_before = sorted((p, p.stat().st_mtime_ns) for p in complete_session.rglob("*"))
    res = runner.invoke(app, ["inventory", str(complete_session), "-w", str(wd)])
    assert res.exit_code == 0, res.output
    assert "2021-09-06" in res.output
    for name in ("inventory.md", "inventory.json", "provenance.json", "config.used.yaml", "logs/inventory.log"):
        assert (wd / name).exists(), name
    data = json.loads((wd / "inventory.json").read_text())
    assert {lv["level"] for lv in data["levels"]} == {"A"}
    prov = json.loads((wd / "provenance.json").read_text())
    assert "astropy" in prov["software"] and "inventory" in prov["steps"]
    md = (wd / "inventory.md").read_text()
    assert "Niveaux de calibration" in md

    res2 = runner.invoke(app, ["inventory", str(complete_session), "-w", str(wd)])
    assert "inchangé" in res2.output
    # Raw files untouched.
    assert sorted((p, p.stat().st_mtime_ns) for p in complete_session.rglob("*")) == raw_before


def test_inventory_reruns_when_files_change(complete_session, tmp_path):
    from fitsgen import write_frame
    wd = tmp_path / "work"
    runner.invoke(app, ["inventory", str(complete_session), "-w", str(wd)])
    write_frame(complete_session / "light/H/new.fits", "LIGHT", 300, "2021-09-07T01:30:00", filt="H")
    res = runner.invoke(app, ["inventory", str(complete_session), "-w", str(wd)])
    assert "inchangé" not in res.output


def test_darks_todo_command(complete_session):
    res = runner.invoke(app, ["darks-todo", str(complete_session)])
    assert res.exit_code == 0
    assert "Aucun dark manquant" in res.output
