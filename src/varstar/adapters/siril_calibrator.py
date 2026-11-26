"""``Calibrator`` with Siril: light_cal = (light − dark) / normalised flat, float32, in ADU16.

No cosmetic correction, no registration, no debayering. Siril writes [0, 1] floats; outputs are
converted back to ADU16 and receive the full raw header plus provenance keywords.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..domain.calibration import CALIB_VERSION, CalibrationSensor
from ..domain.plan import GroupPlan
from .images import NORM_SCALE, read_siril_output, write_float32
from .siril import Siril, link_sequence


def thermal_dark(dark_path: Path, offset_adu: float, out: Path) -> None:
    """Dark minus a uniform offset level, normalised for Siril (needed by -opt=exp scaling)."""
    with fits.open(dark_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32) * np.float32(hdul[0].header.get("VSNORM", 1.0))
        header = hdul[0].header.copy()
    header.remove("VSNORM", ignore_missing=True)
    header["HISTORY"] = f"varstar: thermal dark = master - {offset_adu:.2f} ADU"
    fits.PrimaryHDU(((data - offset_adu) / NORM_SCALE).astype(np.float32), header=header).writeto(
        out, overwrite=True)


def provenance_header(raw: Path, group: GroupPlan, siril_hdr: fits.Header | None,
                      sensor: CalibrationSensor) -> fits.Header:
    header = fits.getheader(raw, 0).copy()
    for key in ("BZERO", "BSCALE"):
        header.remove(key, ignore_missing=True)
    header["BUNIT"] = ("ADU", "ADU16 (raw 16-bit data numbers)")
    header["VSVERS"] = (CALIB_VERSION, "varstar calibration version")
    header["VSRAW"] = str(raw)
    header["VSLEVEL"] = (group.level, "calibration level A/B/C")
    header["VSDARK"] = (group.dark.id if group.dark else "NONE", "master dark id")
    header["VSDKMODE"] = (group.dark.mode if group.dark else "none", "dark mode")
    header["VSFLAT"] = (group.flat.id if group.flat else "NONE", "master flat id")
    header["VSDFLAT"] = (group.darkflat.id if group.darkflat else "NONE", "master dark-flat id")
    header["VSBPM"] = (Path(group.bpm).name if group.bpm else "NONE", "bad pixel map")
    header["VSAMPG"] = (Path(group.ampglow).name if group.ampglow else "NONE", "amp glow mask")
    if sensor.adu_scale:
        header["VSADUSC"] = (sensor.adu_scale, "ADU16 per native ADU (detected)")
    if sensor.effective_gain:
        header["EGAINEFF"] = (sensor.effective_gain, "[e-/ADU16] effective gain = EGAIN / VSADUSC")
    if siril_hdr is not None:
        for card in siril_hdr.get("HISTORY", []):
            header.add_history(f"siril: {card}"[:72])
    return header


def calibrate_command(group: GroupPlan) -> str:
    """Siril ``calibrate`` command of a group (masters linked as mdark/mflat in the work folder)."""
    cmd = "calibrate l"
    if group.dark is not None:
        if group.dark.mode == "scaled":
            cmd += f' -bias="={group.dark.offset_adu:.2f}" -dark=mdark -opt=exp'
        else:
            cmd += " -dark=mdark"
    if group.flat is not None:
        cmd += " -flat=mflat"
    return cmd + " -prefix=pp_"


class SirilCalibrator:
    def __init__(self, siril: Siril, workdir: Path):
        self.siril = siril
        self.workdir = Path(workdir)

    def status(self) -> dict:
        return self.siril.guard_status()

    def calibrate(self, group: GroupPlan, outputs: dict[str, Path], sensor: CalibrationSensor) -> None:
        raws = [Path(p) for p in group.files]
        if group.dark is None and group.flat is None:
            # Level C without any master: keep raw values as float32 (no Siril pass needed).
            for raw in raws:
                data = fits.getdata(raw).astype(np.float32)
                write_float32(outputs[str(raw)], data, provenance_header(raw, group, None, sensor))
            return
        work = self.workdir / "siril" / f"cal_{group.key}"
        if work.exists():
            shutil.rmtree(work)
        link_sequence(raws, work, "l")
        if group.dark is not None:
            if group.dark.mode == "scaled":
                thermal_dark(Path(group.dark.path), group.dark.offset_adu, work / "mdark.fits")
            else:
                (work / "mdark.fits").symlink_to(Path(group.dark.path))
        if group.flat is not None:
            (work / "mflat.fits").symlink_to(Path(group.flat.path))
        self.siril.run([calibrate_command(group)], work, f"calibrate_{group.key}")
        for i, raw in enumerate(raws, 1):
            produced = work / f"pp_l_{i:05d}.fits"
            if not produced.exists():
                raise RuntimeError(f"Siril n'a pas produit {produced.name} pour {raw.name}")
            data, siril_hdr = read_siril_output(produced)
            write_float32(outputs[str(raw)], data, provenance_header(raw, group, siril_hdr, sensor))
        shutil.rmtree(work, ignore_errors=True)
