"""Thin wrapper around siril-cli: generated .ssf scripts, private settings file, captured logs."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED = "1.2.0"


class SirilError(RuntimeError):
    pass


@dataclass
class Siril:
    cli: Path
    workroot: Path  # holds the private settings file
    logs_dir: Path
    _version: str | None = field(default=None, init=False)

    @classmethod
    def from_config(cls, cfg: dict, workdir: Path) -> "Siril":
        cli = Path(os.path.expanduser(cfg["paths"]["siril_cli"]))
        if not cli.exists():
            raise SirilError(f"siril-cli introuvable : {cli} (paramètre paths.siril_cli)")
        return cls(cli, workdir / "siril", workdir / "logs" / "siril")

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

    def run(self, commands: list[str], cwd: Path, name: str, timeout: float = 6 * 3600) -> str:
        """Run a script in `cwd`; raise SirilError (with the failing command) on failure."""
        cwd.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        script = cwd / f"{name}.ssf"
        lines = [f"requires {REQUIRED}", "setext fits", *commands]
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        t0 = time.monotonic()
        proc = subprocess.run([str(self.cli), "-i", str(self.initfile), "-d", str(cwd), "-s", str(script)],
                              capture_output=True, text=True, env=self._env(), timeout=timeout)
        output = proc.stdout + proc.stderr
        log_path = self.logs_dir / f"{name}.log"
        log_path.write_text(
            f"# script ({cwd})\n" + "\n".join(lines) + f"\n# exit={proc.returncode} "
            f"durée={time.monotonic() - t0:.1f}s\n\n" + output, encoding="utf-8")
        if proc.returncode != 0:
            ran = re.findall(r"running command:?\s*(\S+)", output, re.IGNORECASE)
            failing = ran[-1] if ran else "?"
            line = next((ln for ln in lines if ln.split(" ")[0] == failing), failing)
            errors = [ln for ln in output.splitlines() if re.search(r"error|fail|cannot|unable", ln, re.I)]
            detail = errors[-1].strip() if errors else output.strip().splitlines()[-1:]
            raise SirilError(f"Siril a échoué sur la commande « {line} » : {detail} (journal : {log_path})")
        return output


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
