"""Console reporter and per-step log files."""

import logging

from varstar.adapters.console import ConsoleReporter, StepLog


def test_console_reporter_prints(capsys):
    ConsoleReporter().progress("  masters pour sept06 / H …")
    assert capsys.readouterr().out == "  masters pour sept06 / H …\n"


def test_one_log_file_per_step(tmp_path):
    log = StepLog(tmp_path, "INFO")
    log.start("masters")
    logging.getLogger("varstar").info("Construction du master dark")
    log.start("calibrate")
    logging.getLogger("varstar").warning("[C3] fond négatif")
    logging.getLogger("varstar").debug("caché")
    masters = (tmp_path / "logs" / "masters.log").read_text(encoding="utf-8")
    calibrate = (tmp_path / "logs" / "calibrate.log").read_text(encoding="utf-8")
    assert "Construction du master dark" in masters and "C3" not in masters
    assert "WARNING [C3] fond négatif" in calibrate and "caché" not in calibrate
    assert len(logging.getLogger("varstar").handlers) == 1
