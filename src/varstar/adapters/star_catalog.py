"""``StarCatalog`` adapters: Gaia DR3 through TAP services (pyvo, cached on disk), or a local CSV file.

The Gaia cache lives in the shared library (``<library>/catalogs``): a field is downloaded once,
then reused by every run and every target that falls inside the cached cone.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyvo
import requests

from ..domain.astrometry import CatalogStars, angular_sep_arcsec
from ..domain.calibration import PipelineError

log = logging.getLogger("varstar")

GAIA_QUERY = ("SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
              "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.4f})) "
              "AND phot_g_mean_mag < {mag:.2f}")
CACHE_VERSION = 1
COLUMNS = ("source_id", "ra", "dec", "phot_g_mean_mag")
MAXREC = 5_000_000           # far above any field of this pipeline; an overflow is still detected


class _TimeoutSession(requests.Session):
    """A requests session with a default timeout (pyvo has none: a silent server would hang the run)."""

    def __init__(self, timeout_s: float):
        super().__init__()
        self.timeout_s = timeout_s
        self.headers["User-Agent"] = "varstar-pipeline"

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", self.timeout_s)
        return super().request(*args, **kwargs)


def atomic_write(path: Path, text: str) -> None:
    """Write through a uniquely named temporary file then rename: two runs sharing the library never
    see (or produce) a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def tap_problem(exc: Exception, timeout_s: float) -> str:
    """French reason of a failed TAP query.

    pyvo (1.10) never lets a ``requests`` exception escape ``run_sync``: HTTP errors become
    ``DALServiceError``, but a connection failure or a timeout is re-raised as ``DALFormatError``
    with the requests exception as ``cause`` (``execute_votable`` wraps everything). The cause,
    not the class, tells a server down from a real non-VOTable answer (HTML maintenance page).
    """
    cause = getattr(exc, "cause", None)
    if isinstance(cause, requests.Timeout):
        return f"pas de réponse en {timeout_s:g} s"
    if isinstance(cause, requests.ConnectionError):
        return "injoignable (réseau ou serveur arrêté)"
    if isinstance(exc, pyvo.dal.DALFormatError):
        return "réponse inattendue (pas une table VOTable : page de maintenance ?)"
    text = str(getattr(exc, "reason", None) or exc).strip()
    return text.splitlines()[0][:160] if text else type(exc).__name__


def table_to_csv(table: Mapping[str, Any]) -> str:
    """Gaia columns of a result table (astropy Table, or any mapping of columns) as the cache CSV.

    A star is kept only when its identifier, position and magnitude are all present (VOTable mask)
    and finite. Columns of different lengths are refused (``ValueError``).
    """
    cols = [np.ma.asarray(table[c]) for c in COLUMNS]
    if len({c.shape for c in cols}) != 1 or cols[0].ndim != 1:
        raise ValueError("colonnes de longueurs différentes")
    sid = cols[0]
    ra, dec, g = (np.ma.masked_invalid(np.ma.asarray(c, dtype=float)) for c in cols[1:])
    bad = np.ma.getmaskarray(sid) | np.ma.getmaskarray(ra) | np.ma.getmaskarray(dec) | np.ma.getmaskarray(g)
    lines = [",".join(COLUMNS)]
    for i in np.nonzero(~bad)[0]:
        lines.append(f"{int(sid[i])},{float(ra[i]):.9f},{float(dec[i]):.9f},{float(g[i]):.4f}")
    if bad.any():
        log.info("Catalogue Gaia : %d étoile(s) sans position ou sans magnitude G écartée(s)", int(bad.sum()))
    return "\n".join(lines) + "\n"


def _filter(stars: CatalogStars, ra: float, dec: float, radius_deg: float, mag_limit: float,
            source: str) -> CatalogStars:
    sep = angular_sep_arcsec(stars.ra, stars.dec, ra, dec) / 3600
    keep = (sep <= radius_deg) & (stars.mag < mag_limit)
    ids = tuple(i for i, k in zip(stars.ids, keep) if k) if stars.ids else ()
    return CatalogStars(stars.ra[keep], stars.dec[keep], stars.mag[keep], source, ids).sorted_by_mag()


def parse_csv(text: str) -> CatalogStars:
    """CSV with columns ra, dec and a magnitude (mag, g or phot_g_mean_mag); optional source_id/id.

    Rows without magnitude, or with a non-finite position or magnitude (NaN, Inf), are dropped
    (and counted in the log); a non-numeric value or a short row is an error (``ValueError``,
    French message with the line number).
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return CatalogStars(np.empty(0), np.empty(0), np.empty(0))
    keys = {k.strip().lower(): k for k in reader.fieldnames or [] if k is not None}
    try:
        kra, kdec = keys["ra"], keys["dec"]
    except KeyError:
        raise ValueError("colonnes ra et dec attendues") from None
    kmag = next((keys[k] for k in ("mag", "g", "phot_g_mean_mag", "gmag") if k in keys), None)
    if kmag is None:
        raise ValueError("colonne de magnitude attendue (mag, g ou phot_g_mean_mag)")
    kid = next((keys[k] for k in ("source_id", "id") if k in keys), None)
    ra, dec, mag, ids = [], [], [], []
    dropped = 0
    for n, r in enumerate(rows, start=2):          # line 1 is the header
        if any(r.get(k) is None for k in (kra, kdec, kmag)):    # short row (DictReader fills with None)
            raise ValueError(f"ligne {n} illisible (colonnes manquantes) : {dict(r)}")
        if r[kmag].strip() == "":
            dropped += 1
            continue
        try:
            values = (float(r[kra]), float(r[kdec]), float(r[kmag]))
        except (TypeError, ValueError):
            raise ValueError(f"ligne {n} illisible : {dict(r)}") from None
        if not all(math.isfinite(v) for v in values):
            dropped += 1
            continue
        ra.append(values[0])
        dec.append(values[1])
        mag.append(values[2])
        if kid:
            ids.append(str(r[kid]))
    if dropped:
        log.info("Catalogue : %d ligne(s) sans magnitude ou avec une valeur non finie écartée(s)", dropped)
    return CatalogStars(np.array(ra, dtype=float), np.array(dec, dtype=float), np.array(mag, dtype=float),
                        ids=tuple(ids))


class CsvStarCatalog:
    """Stars from a local CSV file (tests, offline use, any other catalogue exported to CSV)."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()

    def describe(self) -> str:
        return f"fichier {self.path.name}"

    def cone(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars:
        try:
            stars = parse_csv(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"catalogue {self.path} illisible : {exc}") from None
        return _filter(stars, ra, dec, radius_deg, mag_limit, self.describe())


class GaiaTapCatalog:
    """Gaia DR3 by ADQL on TAP services (tried in order, through pyvo), cached in ``cache_dir``.

    ``fetch(url, query)`` returns ``(table, overflow)``; it is injectable so that the tests need no
    network. pyvo errors (service down, an HTML page instead of a VOTable, ADQL error) make the
    next service be tried.
    """

    def __init__(self, cache_dir: Path, tap_urls: Sequence[str], timeout_s: float = 300.0,
                 fetch: Callable[[str, str], tuple[Mapping[str, Any], bool]] | None = None):
        self.cache_dir = Path(cache_dir).expanduser()
        self.tap_urls = list(tap_urls)
        self.timeout_s = timeout_s
        self._fetch = fetch or self._tap

    def describe(self) -> str:
        return "Gaia DR3"

    # ------------------------------------------------------------------ cache

    @staticmethod
    def _meta(path: Path) -> dict[str, Any] | None:
        """Metadata of a cache entry, or None when it is unreadable, of another version or malformed."""
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("version") != CACHE_VERSION:
            return None
        try:
            out = {k: float(meta[k]) for k in ("ra", "dec", "radius_deg", "mag_limit")}
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in out.values()):
            return None
        return {**out, "service": str(meta.get("service", "?"))}

    def _cached(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars | None:
        """A cached cone containing the requested one, at least as deep.

        The cache is only an optimisation: an entry that cannot be read (malformed metadata,
        missing or corrupt CSV) is skipped with a warning, and the field is downloaded again.
        """
        if not self.cache_dir.exists():
            return None
        for meta_path in sorted(self.cache_dir.glob("gaia_dr3_*.json")):
            meta = self._meta(meta_path)
            if meta is None:
                log.warning("Cache Gaia : entrée %s illisible, ignorée", meta_path.name)
                continue
            sep = float(angular_sep_arcsec(meta["ra"], meta["dec"], ra, dec)) / 3600
            if sep + radius_deg > meta["radius_deg"] + 1e-6 or meta["mag_limit"] < mag_limit:
                continue
            csv_path = meta_path.with_suffix(".csv")
            try:
                stars = parse_csv(csv_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("Cache Gaia : %s illisible (%s), ignoré", csv_path.name, exc)
                continue
            log.info("Catalogue Gaia en cache : %s", csv_path.name)
            return _filter(stars, ra, dec, radius_deg, mag_limit, f"Gaia DR3 (cache, {meta['service']})")
        return None

    def _store(self, text: str, ra: float, dec: float, radius_deg: float, mag_limit: float, service: str) -> bool:
        """Cache a downloaded cone. A write failure (disk full, read-only library) is logged and
        ignored: the stars just downloaded are still used. The CSV is written before the metadata,
        both atomically, so that an entry is never visible without its complete CSV."""
        key = hashlib.sha256(f"{ra:.4f}|{dec:.4f}|{radius_deg:.3f}|{mag_limit:.2f}".encode()).hexdigest()[:12]
        base = self.cache_dir / f"gaia_dr3_{key}"
        meta = {"version": CACHE_VERSION, "ra": ra, "dec": dec, "radius_deg": radius_deg, "mag_limit": mag_limit,
                "service": service}
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write(base.with_suffix(".csv"), text)
            atomic_write(base.with_suffix(".json"), json.dumps(meta, indent=2))
        except OSError as exc:
            log.warning("Cache Gaia : écriture impossible dans %s (%s) — catalogue utilisé sans cache",
                        self.cache_dir, exc)
            return False
        return True

    # ------------------------------------------------------------------ network

    def _tap(self, url: str, query: str) -> tuple[Mapping[str, Any], bool]:
        service = pyvo.dal.TAPService(url, session=_TimeoutSession(self.timeout_s))
        result = service.run_sync(query, maxrec=MAXREC)
        return result.to_table(), result.query_status == "OVERFLOW"

    def cone(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars:
        cached = self._cached(ra, dec, radius_deg, mag_limit)
        if cached is not None:
            return cached
        query = GAIA_QUERY.format(ra=ra, dec=dec, radius=radius_deg, mag=mag_limit)
        problems = []
        for url in self.tap_urls:
            host = urllib.parse.urlparse(url).netloc or url
            try:
                table, overflow = self._fetch(url, query)
            except pyvo.dal.DALAccessError as exc:
                problems.append(f"{host} : {tap_problem(exc, self.timeout_s)}")
                continue
            if overflow:
                problems.append(f"{host} : résultat tronqué par le service (réduire astrometry.catalog_mag_limit)")
                continue
            try:
                text = table_to_csv(table)
            except (KeyError, ValueError, TypeError) as exc:
                problems.append(f"{host} : colonnes inattendues ({exc})")
                continue
            self._store(text, ra, dec, radius_deg, mag_limit, host)
            log.info("Catalogue Gaia téléchargé depuis %s", host)
            return _filter(parse_csv(text), ra, dec, radius_deg, mag_limit, f"Gaia DR3 ({host})")
        raise PipelineError("catalogue Gaia inaccessible — " + " ; ".join(problems)
                            + ". Réessayer plus tard, ou fournir un fichier CSV (astrometry.catalog).")
