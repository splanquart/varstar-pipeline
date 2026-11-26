"""Siril adapter without Siril: a fake siril-cli (shell script) records the generated scripts.

Python init guard = workaround for free-astro/siril#2021,
see https://github.com/splanquart/varstar-pipeline/issues/1.
"""

import json
import logging
import stat
import sys
from pathlib import Path

import pytest

from varstar.config import load_config
from varstar.domain.settings import replace
from varstar.siril import PYTHON_INIT_GUARD, Siril, SirilError, parse_version

# The fake answers --version with $FAKE_VERSION; with FAKE_PYSCRIPT_FAILS=1 a script containing
# pyscript fails the way Siril does ("Error in line 2 ('pyscript'): ...", exit 1).
FAKE_CLI = """#!/bin/sh
if [ "$1" = "--version" ]; then echo "siril $FAKE_VERSION"; exit 0; fi
while [ $# -gt 0 ]; do [ "$1" = "-s" ] && script="$2"; shift; done
cat "$script" >> "$FAKE_RECORD"; echo "----" >> "$FAKE_RECORD"
if [ "$FAKE_PYSCRIPT_FAILS" = "1" ] && grep -q '^pyscript' "$script"; then
  echo "log: Running command: pyscript"
  echo "log: Error in line 2 ('pyscript'): python venv unavailable."
  exit 1
fi
if grep -q '^boom' "$script"; then
  echo "log: Running command: boom"; echo "log: Error in line 4 ('boom'): cannot do it."; exit 1
fi
echo "log: Script execution finished successfully."
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cli = tmp_path / "siril-cli"
    cli.write_text(FAKE_CLI)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.txt"
    monkeypatch.setenv("FAKE_RECORD", str(record))
    monkeypatch.setenv("FAKE_VERSION", "1.4.5")
    monkeypatch.delenv("FAKE_PYSCRIPT_FAILS", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    def make(mode="auto"):
        return Siril(cli, tmp_path / "work" / "siril", tmp_path / "work" / "logs", python_init_guard=mode)

    def scripts():
        text = record.read_text() if record.exists() else ""
        return [s.strip().splitlines() for s in text.split("----") if s.strip()]

    make.scripts = scripts
    make.tmp = tmp_path
    return make


def run_one(siril, name="job"):
    return siril.run(["stat"], siril.workroot.parent / "run", name)


@pytest.mark.parametrize("text, expected", [
    ("1.4.5", (1, 4, 5)), ("1.4.0-beta3", (1, 4, 0)), ("1.2.6", (1, 2, 6)), ("inconnue", None), ("", None),
])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_guard_line_right_after_requires_on_macos_recent_siril(fake):
    run_one(fake())
    [script] = fake.scripts()
    assert script[0].startswith("requires ")
    assert script[1] == PYTHON_INIT_GUARD
    assert script[2:] == ["setext fits", "stat"]


@pytest.mark.parametrize("platform, version", [
    ("linux", "1.4.5"), ("win32", "1.4.5"),       # no launch constraint outside macOS
    ("darwin", "1.4.0"), ("darwin", "1.2.6"),     # constraint added in 1.4.1; no pyscript before 1.4
    ("darwin", "dev-build"),                      # unknown version: do not risk it
])
def test_no_guard_in_auto_mode_when_not_needed(fake, monkeypatch, platform, version):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("FAKE_VERSION", version)
    run_one(fake())
    assert PYTHON_INIT_GUARD not in fake.scripts()[0]


def test_off_removes_guard_and_on_forces_it(fake, monkeypatch):
    run_one(fake("off"))
    monkeypatch.setattr(sys, "platform", "linux")
    run_one(fake("on"))
    off, on = fake.scripts()
    assert PYTHON_INIT_GUARD not in off
    assert on[1] == PYTHON_INIT_GUARD


def test_invalid_mode_rejected(fake):
    with pytest.raises(SirilError, match="python_init_guard"):
        fake("sometimes")


def test_guard_logged_in_siril_log(fake):
    s = fake()
    run_one(s, "masterdark")
    assert PYTHON_INIT_GUARD in (s.logs_dir / "masterdark.log").read_text()


def test_failing_guard_falls_back_once_and_is_logged(fake, monkeypatch, caplog):
    """Siril's Python broken: rerun without the line, warn, keep it off for the rest of the run."""
    monkeypatch.setenv("FAKE_PYSCRIPT_FAILS", "1")
    s = fake()
    with caplog.at_level(logging.WARNING, logger="varstar"):
        run_one(s, "a")
        run_one(s, "b")
    first, retry, second = fake.scripts()
    assert PYTHON_INIT_GUARD in first
    assert PYTHON_INIT_GUARD not in retry and PYTHON_INIT_GUARD not in second
    assert [r for r in caplog.records if "contournement désactivé" in r.getMessage()]
    assert (s.logs_dir / "a.pyscript-echec.log").exists()
    assert PYTHON_INIT_GUARD not in (s.logs_dir / "a.log").read_text()
    status = s.guard_status()
    assert status["active"] is False and "a.pyscript-echec.log" in status["fallback"]


def test_other_failures_are_not_retried(fake):
    s = fake()
    with pytest.raises(SirilError, match="« boom »"):
        s.run(["boom"], s.workroot.parent / "run", "job")
    assert len(fake.scripts()) == 1
    assert s.guard_status()["fallback"] is None


def test_guard_status_and_config(fake, tmp_path):
    assert fake().guard_status() == {"mode": "auto", "active": True, "fallback": None,
                                     "issue": "https://github.com/splanquart/varstar-pipeline/issues/1"}
    cfg = load_config(overrides={"paths": {"siril_cli": str(fake.tmp / "siril-cli")}})
    assert cfg.siril.python_init_guard == "auto"
    off = replace(cfg, siril={"python_init_guard": "off"})
    assert Siril.from_config(off, tmp_path / "w").python_init_guard == "off"
    json.dumps(fake().guard_status())  # provenance.json must be able to store it


@pytest.mark.parametrize("yaml_value, mode", [("off", "off"), ("on", "on"), ('"off"', "off"), ("auto", "auto")])
def test_unquoted_yaml_on_off(fake, tmp_path, yaml_value, mode):
    """YAML reads unquoted on/off as booleans: they must still mean on/off."""
    user = tmp_path / "config.yaml"
    user.write_text(f"paths:\n  siril_cli: {fake.tmp / 'siril-cli'}\nsiril:\n  python_init_guard: {yaml_value}\n")
    assert Siril.from_config(load_config(user), tmp_path / "w").python_init_guard == mode
