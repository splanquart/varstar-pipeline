"""Light calibration with Siril: light_cal = (light - dark) / normalised flat, float32, in ADU16.

No cosmetic correction, no registration, no debayering. Siril writes [0, 1] floats; outputs are
converted back to ADU16 and receive the full raw header plus provenance keywords.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .imageio import NORM_SCALE, read_siril_output, write_float32
from .siril import Siril, link_sequence

log = logging.getLogger("varstar")

CALIB_VERSION = 1


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", Path(name).stem).strip("_")


def output_dir(workdir: Path, group: dict[str, Any]) -> Path:
    sess = group["session"] + (f"_{group['label']}" if group.get("label") else "")
    return workdir / "calibrated" / sess / group["filter"]


def _fingerprint(group: dict[str, Any]) -> str:
    h = hashlib.sha256()
    payload = {k: group.get(k) for k in ("level", "dark", "flat", "darkflat", "bpm", "ampglow")}
    h.update(json.dumps({"v": CALIB_VERSION, **payload}, sort_keys=True, default=str).encode())
    for raw in group["files"]:
        st = Path(raw).stat()
        h.update(f"{raw}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _thermal_dark(dark_path: Path, offset_adu: float, out: Path) -> None:
    """Dark minus a uniform offset level, normalised for Siril (needed by -opt=exp scaling)."""
    with fits.open(dark_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32) * np.float32(hdul[0].header.get("VSNORM", 1.0))
        header = hdul[0].header.copy()
    header.remove("VSNORM", ignore_missing=True)
    header["HISTORY"] = f"varstar: thermal dark = master - {offset_adu:.2f} ADU"
    fits.PrimaryHDU(((data - offset_adu) / NORM_SCALE).astype(np.float32), header=header).writeto(
        out, overwrite=True)


def _provenance_header(raw: Path, group: dict[str, Any], siril_hdr: fits.Header | None,
                       sensor: dict[str, Any]) -> fits.Header:
    header = fits.getheader(raw, 0).copy()
    for key in ("BZERO", "BSCALE"):
        header.remove(key, ignore_missing=True)
    header["BUNIT"] = ("ADU", "ADU16 (raw 16-bit data numbers)")
    header["VSVERS"] = (CALIB_VERSION, "varstar calibration version")
    header["VSRAW"] = str(raw)
    header["VSLEVEL"] = (group["level"], "calibration level A/B/C")
    header["VSDARK"] = (group["dark"]["id"] if group.get("dark") else "NONE", "master dark id")
    header["VSDKMODE"] = (group["dark"]["mode"] if group.get("dark") else "none", "dark mode")
    header["VSFLAT"] = (group["flat"]["id"] if group.get("flat") else "NONE", "master flat id")
    header["VSDFLAT"] = (group["darkflat"]["id"] if group.get("darkflat") else "NONE", "master dark-flat id")
    header["VSBPM"] = (Path(group["bpm"]).name if group.get("bpm") else "NONE", "bad pixel map")
    header["VSAMPG"] = (Path(group["ampglow"]).name if group.get("ampglow") else "NONE", "amp glow mask")
    if sensor.get("adu_scale"):
        header["VSADUSC"] = (sensor["adu_scale"], "ADU16 per native ADU (detected)")
    if sensor.get("effective_gain"):
        header["EGAINEFF"] = (sensor["effective_gain"], "[e-/ADU16] effective gain = EGAIN / VSADUSC")
    if siril_hdr is not None:
        for card in siril_hdr.get("HISTORY", []):
            header.add_history(f"siril: {card}"[:72])
    return header


def calibrate_group(group: dict[str, Any], workdir: Path, siril: Siril, sensor: dict[str, Any],
                    force: bool = False) -> dict[str, Any]:
    """Calibrate one light group; returns {raw path: calibrated path} plus status."""
    out_dir = output_dir(workdir, group)
    manifest_path = workdir / "manifests" / "calibrate" / f"{group['key']}.json"
    fp = _fingerprint(group)
    if not force and manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prev.get("fingerprint") == fp and all(Path(p).exists() for p in prev["outputs"].values()):
            log.info("Calibration %s inchangée, sautée", group["key"])
            return {**prev, "skipped": True}

    raws = [Path(p) for p in group["files"]]
    outputs: dict[str, str] = {}
    work = workdir / "siril" / f"cal_{group['key']}"
    if work.exists():
        shutil.rmtree(work)

    dark, flat = group.get("dark"), group.get("flat")
    if dark is None and flat is None:
        # Level C without any master: keep raw values as float32 (no Siril pass needed).
        for raw in raws:
            data = fits.getdata(raw).astype(np.float32)
            out = out_dir / f"{safe_stem(raw.name)}.fits"
            write_float32(out, data, _provenance_header(raw, group, None, sensor))
            outputs[str(raw)] = str(out)
    else:
        link_sequence(raws, work, "l")
        cmd = "calibrate l"
        if dark is not None:
            if dark["mode"] == "scaled":
                _thermal_dark(Path(dark["path"]), dark["offset_adu"], work / "mdark.fits")
                cmd += f' -bias="={dark["offset_adu"]:.2f}" -dark=mdark -opt=exp'
            else:
                (work / "mdark.fits").symlink_to(Path(dark["path"]))
                cmd += " -dark=mdark"
        if flat is not None:
            (work / "mflat.fits").symlink_to(Path(flat["path"]))
            cmd += " -flat=mflat"
        cmd += " -prefix=pp_"
        siril.run([cmd], work, f"calibrate_{group['key']}")
        for i, raw in enumerate(raws, 1):
            produced = work / f"pp_l_{i:05d}.fits"
            if not produced.exists():
                raise RuntimeError(f"Siril n'a pas produit {produced.name} pour {raw.name}")
            data, siril_hdr = read_siril_output(produced)
            out = out_dir / f"{safe_stem(raw.name)}.fits"
            write_float32(out, data, _provenance_header(raw, group, siril_hdr, sensor))
            outputs[str(raw)] = str(out)
        shutil.rmtree(work, ignore_errors=True)

    result = {"key": group["key"], "fingerprint": fp, "outputs": outputs, "level": group["level"]}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Calibration %s : %d images", group["key"], len(outputs))
    return {**result, "skipped": False}
