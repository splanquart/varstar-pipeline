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


def test_check_command(tmp_path):
    from fitsgen import two_nights

    root = two_nights(tmp_path / "Y", level2=541.0)
    res = runner.invoke(app, ["check", str(root), "-w", str(tmp_path / "w")])
    assert res.exit_code == 0, res.output
    assert "P1" in res.output and "offset" in res.output
    md = (tmp_path / "w" / "qa" / "qa.md").read_text()
    assert "Niveau des darks par nuit" in md and "Conseil pour la prochaine séance" in md
    res = runner.invoke(app, ["check", str(root), "-w", str(tmp_path / "w"), "--headers-only"])
    assert res.exit_code == 0 and "[P1" not in res.output


def test_invalid_config_is_explained(complete_session, tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("matching:\n  temp_tolerance_c: chaud\n", encoding="utf-8")
    for cmd in ("inventory", "check", "darks-todo", "run"):
        res = runner.invoke(app, [cmd, str(complete_session), "-c", str(bad)])
        assert res.exit_code == 1, cmd
        assert "ERREUR : Configuration invalide : matching.temp_tolerance_c = 'chaud' (nombre attendu)" in res.output
