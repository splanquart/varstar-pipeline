"""Interpretation of N.I.N.A. FITS keywords (header → Frame). Pure: works on any mapping.

Folder names only give labels (SESSION_xxx); the frame type always comes from the header.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .model import BIAS, DARK, DARKFLAT, FLAT, LIGHT, UNKNOWN, Frame

_IMAGETYP_MAP = {
    "LIGHT": LIGHT, "SCIENCE": LIGHT, "OBJECT": LIGHT,
    "DARK": DARK,
    "FLAT": FLAT, "FLATFIELD": FLAT, "SKYFLAT": FLAT, "DOMEFLAT": FLAT,
    "BIAS": BIAS, "OFFSET": BIAS, "ZERO": BIAS,
    "DARKFLAT": DARKFLAT, "FLATDARK": DARKFLAT,
}

_SESSION_DIR_RE = re.compile(r"^session[_\- ]?(.+)$", re.IGNORECASE)


def normalize_imagetyp(value: Any) -> str:
    """Map IMAGETYP variants ('Light Frame', 'Dark Flat', 'FLAT DARK'...) to a canonical type."""
    if value is None:
        return UNKNOWN
    text = str(value).upper().strip()
    if "MASTER" in text:
        return UNKNOWN
    text = re.sub(r"\bFRAMES?\b", "", text)
    text = re.sub(r"[\s_\-]+", "", text)
    return _IMAGETYP_MAP.get(text, UNKNOWN)


def _num(header: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            try:
                return float(header[key])
            except (TypeError, ValueError):
                continue
    return None


def _str(header: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return str(header[key]).strip()
    return None


def parse_date_obs(value: Any) -> dt.datetime | None:
    """Parse a FITS DATE-OBS (UTC) into a naive UTC datetime."""
    if value in (None, ""):
        return None
    text = str(value).strip().rstrip("Z")
    # Python's fromisoformat accepts up to 6 fractional digits.
    m = re.match(r"^(.*?\.\d{1,6})\d*$", text)
    if m:
        text = m.group(1)
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def folder_label_from_path(rel_path: str) -> str | None:
    """Return the label of the innermost SESSION_xxx folder, if any (label only, never used for typing)."""
    label = None
    for part in Path(rel_path).parent.parts:
        m = _SESSION_DIR_RE.match(part)
        if m:
            label = m.group(1)
    return label


def frame_from_header(path: Path, header: Mapping[str, Any], root: Path) -> Frame:
    """Build a Frame from FITS keywords (any mapping: astropy Header, dict…). No file access.

    Header problems that make the frame unusable are listed in ``Frame.issues``.
    """
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    imagetyp_raw = _str(header, "IMAGETYP", "FRAMETYP", "FRAME")
    frame = Frame(
        path=path,
        rel_path=rel,
        imagetyp=normalize_imagetyp(imagetyp_raw),
        imagetyp_raw=imagetyp_raw,
        exptime=_num(header, "EXPTIME", "EXPOSURE"),
        gain=_num(header, "GAIN"),
        offset=_num(header, "OFFSET"),
        xbin=int(_num(header, "XBINNING") or 1),
        ybin=int(_num(header, "YBINNING") or 1),
        naxis1=int(header.get("NAXIS1", 0)),
        naxis2=int(header.get("NAXIS2", 0)),
        bitpix=int(header.get("BITPIX", 0)),
        instrument=_str(header, "INSTRUME") or "?",
        ccd_temp=_num(header, "CCD-TEMP", "CCD_TEMP"),
        set_temp=_num(header, "SET-TEMP", "SET_TEMP"),
        filter=_str(header, "FILTER"),
        date_obs=parse_date_obs(header.get("DATE-OBS")),
        object=_str(header, "OBJECT"),
        ra=_num(header, "RA"),
        dec=_num(header, "DEC"),
        focallen=_num(header, "FOCALLEN"),
        xpixsz=_num(header, "XPIXSZ"),
        egain=_num(header, "EGAIN"),
        sitelat=_num(header, "SITELAT"),
        sitelong=_num(header, "SITELONG"),
        siteelev=_num(header, "SITEELEV"),
        airmass=_num(header, "AIRMASS"),
        telescope=_str(header, "TELESCOP"),
        swcreate=_str(header, "SWCREATE"),
        focpos=_num(header, "FOCPOS", "FOCUSPOS"),
        foctemp=_num(header, "FOCTEMP", "FOCUSTEM"),
        centalt=_num(header, "CENTALT"),
        cloudcvr=_num(header, "CLOUDCVR"),
        ambtemp=_num(header, "AMBTEMP"),
        dewpoint=_num(header, "DEWPOINT"),
        humidity=_num(header, "HUMIDITY"),
        folder_label=folder_label_from_path(rel),
    )
    if frame.imagetyp == UNKNOWN:
        frame.issues.append(f"IMAGETYP non reconnu : {imagetyp_raw!r}")
    if frame.exptime is None:
        frame.issues.append("EXPTIME/EXPOSURE absent")
    if frame.date_obs is None:
        frame.issues.append("DATE-OBS absent ou illisible")
    if frame.naxis1 == 0 or frame.naxis2 == 0:
        frame.issues.append("dimensions absentes (NAXIS1/NAXIS2)")
    return frame
