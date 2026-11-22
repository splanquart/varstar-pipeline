"""Image I/O helpers: Siril-normalised floats <-> ADU, minimal XISF reader (PixInsight masters)."""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from astropy.io import fits

# Siril (and PixInsight) store float images normalised to [0, 1] for 16-bit input data.
NORM_SCALE = 65535.0


def read_adu(path: Path, norm: float | None = None) -> tuple[np.ndarray, fits.Header]:
    """Read a FITS image in ADU16 units (float32).

    The scale comes from the VSNORM keyword (varstar masters) or from `norm` (raw Siril output,
    whose floats are normalised to [0, 1]); otherwise the data are returned as stored.
    """
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    scale = float(header.get("VSNORM", norm or 1.0))
    if scale != 1.0:
        data = data * np.float32(scale)
    return data, header


def read_siril_output(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Read a float image written by Siril from 16-bit inputs and convert it back to ADU16."""
    data, header = read_adu(path, norm=None)
    finite = data[np.isfinite(data)]
    if finite.size and float(np.max(np.abs(finite))) > 1.5:
        raise ValueError(f"{path.name} : données Siril hors de l'intervalle normalisé attendu [0, 1]")
    return data * np.float32(NORM_SCALE), header


def write_float32(path: Path, data: np.ndarray, header: fits.Header | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=header)
    for key in ("BZERO", "BSCALE"):
        hdu.header.remove(key, ignore_missing=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    hdu.writeto(tmp, overwrite=True, output_verify="silentfix")
    tmp.replace(path)


def write_mask(path: Path, mask: np.ndarray, header: fits.Header | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.asarray(mask, dtype=np.uint8), header=header)
    tmp = path.with_suffix(path.suffix + ".tmp")
    hdu.writeto(tmp, overwrite=True)
    tmp.replace(path)


class XisfError(ValueError):
    pass


def read_xisf(path: Path) -> np.ndarray:
    """Read the first monochrome, uncompressed image of an XISF file (values as stored)."""
    with open(path, "rb") as fh:
        if fh.read(8) != b"XISF0100":
            raise XisfError("signature XISF absente")
        (hlen,) = struct.unpack("<I", fh.read(4))
        fh.read(4)
        xml = fh.read(hlen).decode("utf-8", errors="replace")
        root = ET.fromstring(xml)
        ns = {"x": "http://www.pixinsight.com/xisf"}
        img = root.find("x:Image", ns)
        if img is None:
            raise XisfError("aucune image dans le fichier XISF")
        if img.get("compression"):
            raise XisfError(f"image XISF compressée ({img.get('compression')}) non prise en charge")
        w, h, c = (int(x) for x in img.get("geometry").split(":"))
        if c != 1:
            raise XisfError("image XISF non monochrome")
        fmt = {"Float32": "f4", "Float64": "f8", "UInt16": "u2", "UInt32": "u4", "UInt8": "u1"}[
            img.get("sampleFormat")]
        order = "<" if img.get("byteOrder", "little") == "little" else ">"
        m = re.match(r"attachment:(\d+):(\d+)", img.get("location", ""))
        if not m:
            raise XisfError("emplacement des données XISF non pris en charge")
        offset, size = int(m.group(1)), int(m.group(2))
        fh.seek(offset)
        data = np.frombuffer(fh.read(size), dtype=order + fmt)
    return data.reshape(h, w)
