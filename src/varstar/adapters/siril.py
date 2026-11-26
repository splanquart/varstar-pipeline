"""Thin wrapper around siril-cli: generated .ssf scripts, private settings file, captured logs."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.settings import Settings

REQUIRED = "1.2.0"

# Workaround for free-astro/siril#2021 (https://github.com/splanquart/varstar-pipeline/issues/1):
# on macOS, siril-cli exits before its background Python venv check has finished; the bundled
# Python, orphaned under launchd, then violates its launch constraint and is killed (crash dialog).
# `pyscript /dev/null` waits for that check (synchronous in scripts since free-astro/siril!1090).
PYTHON_INIT_GUARD = "pyscript /dev/null"
GUARD_MIN_VERSION = (1, 4, 1)  # launch constraint added in 1.4.1 (free-astro/siril#1957)
GUARD_MODES = ("auto", "on", "off")

log = logging.getLogger("varstar")


def parse_version(text: str) -> tuple[int, int, int] | None:
    """'1.4.5' / '1.4.0-beta3' -> (1, 4, 5) / (1, 4, 0); None if unparsable."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


class SirilError(RuntimeError):
    pass


@dataclass
class Siril:
    cli: Path
    workroot: Path  # holds the private settings file
    logs_dir: Path
    python_init_guard: str = "auto"  # auto | on | off (see PYTHON_INIT_GUARD)
    _version: str | None = field(default=None, init=False)
    _guard_fallback: str | None = field(default=None, init=False)  # why the guard was dropped

    def __post_init__(self) -> None:
        if self.python_init_guard not in GUARD_MODES:
            raise SirilError(f"siril.python_init_guard invalide : « {self.python_init_guard} » "
                             f"(valeurs possibles : {', '.join(GUARD_MODES)})")

    @classmethod
    def from_config(cls, cfg: Settings, workdir: Path) -> "Siril":
        cli = Path(os.path.expanduser(cfg.paths.siril_cli))
        if not cli.exists():
            raise SirilError(f"siril-cli introuvable : {cli} (paramètre paths.siril_cli)")
        guard = cfg.siril.python_init_guard   # on/off booleans already normalised by the settings
        return cls(cli, workdir / "siril", workdir / "logs" / "siril", python_init_guard=guard)

    @property
    def initfile(self) -> Path:
        """Private settings file: never touch the user's own Siril preferences."""
        path = self.workroot / "siril.ini"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[core]\nforce_16bit=false\nextension=.fits\n", encoding="utf-8")
        return path

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(LANGUAGE="en", LC_ALL="en_US.UTF-8", LANG="en_US.UTF-8")
        return env

    def version(self) -> str:
        if self._version is None:
            out = subprocess.run([str(self.cli), "--version"], capture_output=True, text=True,
                                 env=self._env(), timeout=60)
            m = re.search(r"siril\s+(\d+\.\d+\.\d+\S*)", out.stdout + out.stderr, re.IGNORECASE)
            self._version = m.group(1) if m else "inconnue"
        return self._version

    def guard_active(self) -> bool:
        """Whether scripts get the PYTHON_INIT_GUARD line (free-astro/siril#2021 workaround)."""
        if self._guard_fallback is not None or self.python_init_guard == "off":
            return False
        if self.python_init_guard == "on":
            return True
        version = parse_version(self.version())
        return sys.platform == "darwin" and version is not None and version >= GUARD_MIN_VERSION

    def guard_status(self) -> dict[str, object]:
        """Workaround state, for provenance.json."""
        return {"mode": self.python_init_guard, "active": self.guard_active(),
                "fallback": self._guard_fallback,
                "issue": "https://github.com/splanquart/varstar-pipeline/issues/1"}

    def run(self, commands: list[str], cwd: Path, name: str, timeout: float = 6 * 3600) -> str:
        """Run a script in `cwd`; raise SirilError (with the failing command) on failure."""
        cwd.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        guard = self.guard_active()
        lines, proc, output, log_path = self._attempt(commands, cwd, name, timeout, guard)
        if proc.returncode != 0 and guard and self._failing_command(output) == "pyscript":
            # Siril's own Python environment is broken: never fail the pipeline for a cosmetic
            # workaround, but never drop it silently either.
            failed_log = log_path.with_name(f"{name}.pyscript-echec.log")
            log_path.replace(failed_log)
            self._guard_fallback = (f"« {PYTHON_INIT_GUARD} » a échoué (journal : {failed_log}) ; "
                                    "contournement désactivé pour le reste de l'exécution")
            log.warning("Siril : %s. Les fenêtres « Python a quitté de manière imprévue » peuvent "
                        "réapparaître (https://github.com/splanquart/varstar-pipeline/issues/1).",
                        self._guard_fallback)
            lines, proc, output, log_path = self._attempt(commands, cwd, name, timeout, False)
        if proc.returncode != 0:
            failing = self._failing_command(output)
            line = next((ln for ln in lines if ln.split(" ")[0] == failing), failing)
            errors = [ln for ln in output.splitlines() if re.search(r"error|fail|cannot|unable", ln, re.I)]
            detail = errors[-1].strip() if errors else output.strip().splitlines()[-1:]
            raise SirilError(f"Siril a échoué sur la commande « {line} » : {detail} (journal : {log_path})")
        return output

    def _attempt(self, commands: list[str], cwd: Path, name: str, timeout: float, guard: bool):
        script = cwd / f"{name}.ssf"
        lines = [f"requires {REQUIRED}", *([PYTHON_INIT_GUARD] if guard else []), "setext fits", *commands]
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        t0 = time.monotonic()
        proc = subprocess.run([str(self.cli), "-i", str(self.initfile), "-d", str(cwd), "-s", str(script)],
                              capture_output=True, text=True, env=self._env(), timeout=timeout)
        output = proc.stdout + proc.stderr
        log_path = self.logs_dir / f"{name}.log"
        log_path.write_text(
            f"# script ({cwd})\n" + "\n".join(lines) + f"\n# exit={proc.returncode} "
            f"durée={time.monotonic() - t0:.1f}s\n\n" + output, encoding="utf-8")
        return lines, proc, output, log_path

    @staticmethod
    def _failing_command(output: str) -> str:
        # "Error in line 2 ('pyscript'): ..." also names unknown commands, which never "run".
        named = re.findall(r"line \d+ \('([^']+)'\)", output, re.IGNORECASE)
        if named:
            return named[-1]
        ran = re.findall(r"running command:?\s*(\S+)", output, re.IGNORECASE)
        return ran[-1] if ran else "?"


def link_sequence(paths: list[Path], directory: Path, base: str) -> list[Path]:
    """Create Siril-friendly symlinks <base>_00001.fits ... (raw names may contain spaces)."""
    directory.mkdir(parents=True, exist_ok=True)
    links = []
    for i, src in enumerate(paths, 1):
        link = directory / f"{base}_{i:05d}.fits"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(src).resolve())
        links.append(link)
    seq = directory / f"{base}_.seq"
    if seq.exists():
        seq.unlink()
    return links
