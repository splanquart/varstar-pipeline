"""Capture and compare the pipeline outputs of a work directory (refactoring safety net).

Two uses:
- ``capture <workdir>``: snapshot of the text outputs (inventory.json, masters/plan.json,
  qa/qa.json) plus a compact signature of every master, mask and calibrated image
  (no image is copied) into ``<workdir>/_baseline/``.
- ``compare <baseline> <workdir>``: readable diff between a snapshot and the current
  outputs, ignoring timestamps, absolute paths and master ids (aliased by content).

The normalisation helpers are also used by the golden tests (tests/e2e).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

VOLATILE_KEYS = {"created", "updated", "generated", "siril_version"}
TEXT_OUTPUTS = ("inventory.json", "masters/plan.json", "qa/qa.json")
ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])")
SHORT_ID_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
NOW_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
NOW_FR_RE = re.compile(r"(\d{2}/\d{2}/\d{4}) \d{2}:\d{2}(?::\d{2})?")
REL_TOL = 1e-5
ABS_TOL = 1e-6


# --------------------------------------------------------------------------- normalisation


def master_aliases(plan: dict[str, Any]) -> dict[str, str]:
    """Map each 16-hex master id of a masters plan to a stable, content-based alias.

    Master ids hash absolute paths and mtimes, so they change with the location of the data.
    The alias is ``<label>@<DATE-OBS of the first source frame>``, read from master.json.
    """
    aliases: dict[str, str] = {}
    for mid, info in (plan.get("masters") or {}).items():
        label = info.get("label") or info.get("kind", "master")
        first = None
        meta_path = Path(info.get("dir", "")) / "master.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dates = [s.get("date_obs") for s in meta.get("frame_stats", []) if s.get("date_obs")]
            first = min(dates)[:16] if dates else None
        aliases[mid] = f"<{label}@{first or '?'}>"
    return aliases


def _recent_dates() -> set[str]:
    """Today and yesterday, ISO and French formats (a run may straddle midnight)."""
    days = [dt.date.today() - dt.timedelta(days=d) for d in (0, 1)]
    return {d.isoformat() for d in days} | {d.strftime("%d/%m/%Y") for d in days}


def _norm_str(s: str, replacements: list[tuple[str, str]], aliases: dict[str, str]) -> str:
    """Paths → placeholders, master ids (and their 8-char prefixes) → aliases, run time → <NOW>.

    Only timestamps of today or yesterday are treated as run times, so observation dates
    (DATE-OBS, 2021) are kept.
    """
    for old, new in replacements:
        s = s.replace(old, new)
    s = ID_RE.sub(lambda m: aliases.get(m.group(0), "<id>"), s)
    short = {k[:8]: v for k, v in aliases.items()}
    s = SHORT_ID_RE.sub(lambda m: short.get(m.group(0), m.group(0)), s)
    recent = _recent_dates()
    now = lambda m: "<NOW>" if m.group(1) in recent else m.group(0)  # noqa: E731
    return NOW_FR_RE.sub(now, NOW_RE.sub(now, s))


def normalize(obj: Any, replacements: dict[str, str] | None = None,
              aliases: dict[str, str] | None = None) -> Any:
    """Copy of ``obj`` without volatile keys, with paths replaced and master ids aliased."""
    repl = sorted((replacements or {}).items(), key=lambda kv: -len(kv[0]))
    al = aliases or {}

    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            return {_norm_str(str(k), repl, al): walk(v) for k, v in o.items() if k not in VOLATILE_KEYS}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return _norm_str(o, repl, al)
        return o

    return walk(obj)


def normalize_text(text: str, replacements: dict[str, str] | None = None,
                   aliases: dict[str, str] | None = None) -> str:
    repl = sorted((replacements or {}).items(), key=lambda kv: -len(kv[0]))
    return _norm_str(text, repl, aliases or {})


# --------------------------------------------------------------------------- diff


def _close(a: float, b: float, rel: float, abs_: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def diff(expected: Any, actual: Any, rel: float = REL_TOL, abs_: float = ABS_TOL,
         path: str = "") -> list[str]:
    """Differences between two JSON-like structures (floats compared with tolerances)."""
    where = path or "<racine>"
    num = (int, float)
    if isinstance(expected, bool) or isinstance(actual, bool):
        same = isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
        return [] if same else [f"{where} : {expected!r} → {actual!r}"]
    if isinstance(expected, num) and isinstance(actual, num):
        return [] if _close(float(expected), float(actual), rel, abs_) else [
            f"{where} : {expected!r} → {actual!r}"]
    if type(expected) is not type(actual):
        return [f"{where} : type {type(expected).__name__} → {type(actual).__name__}"]
    if isinstance(expected, dict):
        out = []
        for k in sorted(set(expected) | set(actual), key=str):
            if k not in actual:
                out.append(f"{path}.{k} : clé disparue")
            elif k not in expected:
                out.append(f"{path}.{k} : clé nouvelle")
            else:
                out += diff(expected[k], actual[k], rel, abs_, f"{path}.{k}")
        return out
    if isinstance(expected, list):
        out = []
        if len(expected) != len(actual):
            out.append(f"{where} : {len(expected)} éléments → {len(actual)}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            out += diff(a, b, rel, abs_, f"{path}[{i}]")
        return out
    return [] if expected == actual else [f"{where} : {expected!r} → {actual!r}"]


# --------------------------------------------------------------------------- image signatures


def image_signature(path: Path) -> dict[str, Any]:
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
    finite = data[np.isfinite(data)] if data.dtype.kind == "f" else data
    med = float(np.median(finite)) if finite.size else float("nan")
    mad = float(np.median(np.abs(finite - med))) * 1.4826 if finite.size else float("nan")
    return {"shape": list(data.shape), "dtype": str(data.dtype), "median": med, "robust_sigma": mad,
            "n_nonfinite": int(data.size - finite.size), "nonzero": int(np.count_nonzero(data)),
            "sha256": hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()}


def collect_images(workdir: Path, plan: dict[str, Any], aliases: dict[str, str]) -> dict[str, Path]:
    """Masters, masks and calibrated images referenced by the work directory, by stable name."""
    images: dict[str, Path] = {}
    for mid, info in (plan.get("masters") or {}).items():
        p = Path(info.get("dir", "")) / "master.fits"
        if p.exists():
            images[f"master {aliases.get(mid, mid)}"] = p
    for g in (plan.get("groups") or {}).values():
        for kind in ("bpm", "ampglow"):
            if g.get(kind) and Path(g[kind]).exists():
                images[normalize_text(f"{kind} {Path(g[kind]).name}", aliases=aliases)] = Path(g[kind])
    for p in sorted((workdir / "calibrated").rglob("*.fits")):
        images[f"calibrated {p.relative_to(workdir / 'calibrated')}"] = p
    return images


# --------------------------------------------------------------------------- capture / compare


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def capture(workdir: Path, out: Path | None = None) -> Path:
    out = out or workdir / "_baseline"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in TEXT_OUTPUTS:
        src = workdir / rel
        if src.exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)
    plan = _load(workdir / "masters/plan.json") or {}
    aliases = master_aliases(plan)
    images = collect_images(workdir, plan, aliases)
    sigs = {}
    for i, (name, p) in enumerate(sorted(images.items()), 1):
        print(f"  signature {i}/{len(images)} : {name}", flush=True)
        sigs[name] = image_signature(p)
    (out / "meta.json").write_text(json.dumps({"workdir": str(workdir)}, indent=2), encoding="utf-8")
    (out / "aliases.json").write_text(json.dumps(aliases, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "signatures.json").write_text(json.dumps(sigs, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def compare(baseline: Path, workdir: Path, rel: float = REL_TOL, abs_: float = ABS_TOL) -> list[str]:
    problems: list[str] = []
    base_aliases = _load(baseline / "aliases.json") or {}
    base_workdir = (_load(baseline / "meta.json") or {}).get("workdir", str(baseline.parent))
    cur_plan = _load(workdir / "masters/plan.json") or {}
    cur_aliases = master_aliases(cur_plan)
    for rel_path in TEXT_OUTPUTS:
        a, b = _load(baseline / rel_path), _load(workdir / rel_path)
        if a is None and b is None:
            continue
        if a is None or b is None:
            problems.append(f"{rel_path} : présent d'un seul côté")
            continue
        d = diff(normalize(a, {base_workdir: "<WORK>"}, base_aliases),
                 normalize(b, {str(workdir): "<WORK>"}, cur_aliases), rel, abs_)
        problems += [f"{rel_path} {x}" for x in d]

    base_sigs = _load(baseline / "signatures.json") or {}
    cur_images = collect_images(workdir, cur_plan, cur_aliases)
    for name in sorted(set(base_sigs) | set(cur_images)):
        if name not in cur_images:
            problems.append(f"image disparue : {name}")
            continue
        if name not in base_sigs:
            problems.append(f"image nouvelle : {name}")
            continue
        sig = image_signature(cur_images[name])
        if sig["sha256"] == base_sigs[name]["sha256"]:
            continue
        d = diff({k: v for k, v in base_sigs[name].items() if k != "sha256"},
                 {k: v for k, v in sig.items() if k != "sha256"}, rel, abs_)
        problems += [f"image {name} {x}" for x in d] or [f"image {name} : pixels différents, statistiques identiques"]
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="instantané des sorties d'un dossier de travail")
    c.add_argument("workdir", type=Path)
    c.add_argument("--out", type=Path, default=None)
    k = sub.add_parser("compare", help="compare un instantané aux sorties actuelles")
    k.add_argument("baseline", type=Path)
    k.add_argument("workdir", type=Path)
    k.add_argument("--rel", type=float, default=REL_TOL)
    k.add_argument("--abs", dest="abs_", type=float, default=ABS_TOL)
    args = ap.parse_args(argv)
    if args.cmd == "capture":
        out = capture(args.workdir.resolve(), args.out)
        print(f"Instantané écrit dans {out}")
        return 0
    problems = compare(args.baseline.resolve(), args.workdir.resolve(), args.rel, args.abs_)
    if not problems:
        print("Aucune différence (hors horodatages, chemins et identifiants).")
        return 0
    print(f"{len(problems)} différence(s) :")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
