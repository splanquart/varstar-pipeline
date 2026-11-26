"""Console side of the use cases: progress lines for the user and one log file per step."""

from __future__ import annotations

import logging
from pathlib import Path


class ConsoleReporter:
    """``Reporter`` printing progress lines (French) on the standard output."""

    def progress(self, message: str) -> None:
        print(message, flush=True)


class StepLog:
    """Logs of the ``varstar`` logger go to <workdir>/logs/<step>.log (one file per step)."""

    def __init__(self, workdir: Path, level: str = "INFO"):
        self.workdir = Path(workdir)
        self.level = level

    def start(self, step: str) -> None:
        setup_logging(self.workdir, step, self.level)


def setup_logging(workdir: Path, step: str, level: str = "INFO") -> None:
    (workdir / "logs").mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("varstar")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    fh = logging.FileHandler(workdir / "logs" / f"{step}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)
