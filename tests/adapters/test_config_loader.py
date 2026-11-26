"""Configuration files → typed settings, with clear French errors."""

import datetime as dt
from pathlib import Path

import pytest
import yaml

from varstar.adapters.config_loader import load_defaults, load_raw, load_settings, resolve_workdir
from varstar.domain.settings import Settings, SettingsError, replace


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_are_typed():
    cfg = load_settings()
    assert cfg.matching.temp_tolerance_c == 1.5 and isinstance(cfg.qa.offset_jump_adu, float)
    assert cfg.masks.flat_local_block_px == 32 and isinstance(cfg.masks.flat_local_block_px, int)
    assert cfg.inventory.ignore_dirs == ("removed", "output", "*.pxiproject")
    assert cfg.matching.same_optical_setup == () and cfg.paths.pixinsight_masters == "auto"
    assert cfg.paths.workdir is None and cfg.siril.python_init_guard == "auto"
    assert cfg.raw == load_defaults()                      # written as is to config.used.yaml


def test_user_file_overrides_the_defaults(tmp_path):
    cfg = load_settings(write(tmp_path, "matching:\n  same_optical_setup:\n    - [2021-09-02, 2021-09-07]\n"
                                        "qa:\n  pixel_step: 2\n"))
    assert cfg.matching.same_optical_setup == ((dt.date(2021, 9, 2), dt.date(2021, 9, 7)),)
    assert cfg.qa.pixel_step == 2 and cfg.qa.min_darks == 15
    assert cfg.raw["matching"]["same_optical_setup"] == [[dt.date(2021, 9, 2), dt.date(2021, 9, 7)]]
    assert load_settings(overrides={"runtime": {"processes": 1}}).runtime.processes == 1


@pytest.mark.parametrize("text, message", [
    ("photometrie:\n  k: 1\n", "Sections de configuration inconnues"),
    ("qa:\n  pixel_stpe: 2\n", "clé(s) inconnue(s) dans « qa » : pixel_stpe"),
    ("matching:\n  temp_tolerance_c: chaud\n", "matching.temp_tolerance_c = 'chaud' (nombre attendu)"),
    ("masks:\n  flat_local_block_px: 3.5\n", "masks.flat_local_block_px = 3.5 (nombre entier attendu)"),
    ("masks:\n  amp_glow_detect: 1\n", "masks.amp_glow_detect = 1 (true ou false attendu)"),
    ("inventory:\n  ignore_dirs: removed\n", "inventory.ignore_dirs = 'removed' (liste attendue)"),
    ("matching:\n  same_optical_setup:\n    - [2021-09-02]\n", "période invalide"),
    ("matching:\n  same_optical_setup:\n    - [2021-09-02, demain]\n", "date invalide"),
    ("qa: [1, 2\n", "Fichier de configuration illisible"),
    ("- a\n- b\n", "sections attendues"),
])
def test_invalid_files_are_explained_in_french(tmp_path, text, message):
    with pytest.raises(SettingsError, match=message.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        load_settings(write(tmp_path, text))


@pytest.mark.parametrize("value, mode", [("on", "on"), ("off", "off"), ('"off"', "off"), ("auto", "auto")])
def test_unquoted_on_off_mean_on_off(tmp_path, value, mode):
    assert load_settings(write(tmp_path, f"siril:\n  python_init_guard: {value}\n")).siril.python_init_guard == mode


def test_settings_are_frozen_and_replace_checks_again():
    cfg = load_settings()
    with pytest.raises(AttributeError):
        cfg.qa.pixel_step = 1
    other = replace(cfg, qa={"pixel_step": 1})
    assert other.qa.pixel_step == 1 and cfg.qa.pixel_step == 4 and other != cfg
    assert replace(cfg) == cfg
    with pytest.raises(SettingsError):
        replace(cfg, qa={"pixel_step": "un"})


def test_raw_is_a_copy():
    data = load_raw()
    cfg = Settings.from_dict(data)
    data["qa"]["pixel_step"] = 99
    assert cfg.raw["qa"]["pixel_step"] == 4 and cfg.section("qa")["pixel_step"] == 4


def test_missing_section_or_key():
    data = load_raw()
    del data["masks"]
    with pytest.raises(SettingsError, match="section\\(s\\) absente\\(s\\) masks"):
        Settings.from_dict(data)
    data = load_raw()
    del data["qa"]["pixel_step"]
    with pytest.raises(SettingsError, match="qa.pixel_step absent"):
        Settings.from_dict(data)


def test_config_used_yaml_round_trips():
    cfg = load_settings()
    assert Settings.from_dict(yaml.safe_load(yaml.safe_dump(dict(cfg.raw), allow_unicode=True))) == cfg


def test_resolve_workdir(tmp_path):
    cfg = load_settings()
    assert resolve_workdir(tmp_path / "YEAR_2021", cfg) == tmp_path / "varstar_work" / "YEAR_2021"
    assert resolve_workdir(tmp_path / "Y", cfg, tmp_path / "w") == (tmp_path / "w").resolve()
    other = replace(cfg, paths={"workdir": str(tmp_path / "ww")})
    assert resolve_workdir(Path("x"), other) == (tmp_path / "ww").resolve()
