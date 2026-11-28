"""``StarCatalog`` adapters: Gaia DR3 through TAP services (cached on disk), or a local CSV file.

The Gaia cache lives in the shared library (``<library>/catalogs``): a field is downloaded once,
then reused by every run and every target that falls inside the cached cone.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np

from ..domain.astrometry import CatalogStars, angular_sep_arcsec
from ..domain.calibration import PipelineError

log = logging.getLogger("varstar")

GAIA_QUERY = ("SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
              "WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius:.4f})) "
              "AND phot_g_mean_mag < {mag:.2f}")
CACHE_VERSION = 1


def _filter(stars: CatalogStars, ra: float, dec: float, radius_deg: float, mag_limit: float,
            source: str) -> CatalogStars:
    sep = angular_sep_arcsec(stars.ra, stars.dec, ra, dec) / 3600
    keep = (sep <= radius_deg) & (stars.mag < mag_limit)
    ids = tuple(i for i, k in zip(stars.ids, keep) if k) if stars.ids else ()
    return CatalogStars(stars.ra[keep], stars.dec[keep], stars.mag[keep], source, ids).sorted_by_mag()


def parse_csv(text: str) -> CatalogStars:
    """CSV with columns ra, dec and a magnitude (mag, g or phot_g_mean_mag); optional source_id/id."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return CatalogStars(np.empty(0), np.empty(0), np.empty(0))
    keys = {k.strip().lower(): k for k in rows[0]}
    try:
        kra, kdec = keys["ra"], keys["dec"]
    except KeyError:
        raise ValueError("colonnes ra et dec attendues") from None
    kmag = next((keys[k] for k in ("mag", "g", "phot_g_mean_mag", "gmag") if k in keys), None)
    if kmag is None:
        raise ValueError("colonne de magnitude attendue (mag, g ou phot_g_mean_mag)")
    kid = next((keys[k] for k in ("source_id", "id") if k in keys), None)
    rows = [r for r in rows if r[kmag] not in ("", None)]
    return CatalogStars(np.array([float(r[kra]) for r in rows]), np.array([float(r[kdec]) for r in rows]),
                        np.array([float(r[kmag]) for r in rows]),
                        ids=tuple(str(r[kid]) for r in rows) if kid else ())


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
    """Gaia DR3 by ADQL on TAP services (tried in order), cached in ``cache_dir``."""

    def __init__(self, cache_dir: Path, tap_urls: Sequence[str], timeout_s: float = 300.0,
                 fetch=None):
        self.cache_dir = Path(cache_dir).expanduser()
        self.tap_urls = list(tap_urls)
        self.timeout_s = timeout_s
        self._fetch = fetch or self._http

    def describe(self) -> str:
        return "Gaia DR3"

    # ------------------------------------------------------------------ cache

    def _cached(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars | None:
        """A cached cone containing the requested one, at least as deep."""
        if not self.cache_dir.exists():
            return None
        for meta_path in sorted(self.cache_dir.glob("gaia_dr3_*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if meta.get("version") != CACHE_VERSION:
                continue
            sep = float(angular_sep_arcsec(meta["ra"], meta["dec"], ra, dec)) / 3600
            if sep + radius_deg <= meta["radius_deg"] + 1e-6 and meta["mag_limit"] >= mag_limit:
                csv_path = meta_path.with_suffix(".csv")
                if csv_path.exists():
                    log.info("Catalogue Gaia en cache : %s", csv_path.name)
                    return _filter(parse_csv(csv_path.read_text(encoding="utf-8")), ra, dec, radius_deg,
                                   mag_limit, f"Gaia DR3 (cache, {meta['service']})")
        return None

    def _store(self, text: str, ra: float, dec: float, radius_deg: float, mag_limit: float, service: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"{ra:.4f}|{dec:.4f}|{radius_deg:.3f}|{mag_limit:.2f}".encode()).hexdigest()[:12]
        base = self.cache_dir / f"gaia_dr3_{key}"
        tmp = base.with_suffix(".csv.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(base.with_suffix(".csv"))
        base.with_suffix(".json").write_text(json.dumps(
            {"version": CACHE_VERSION, "ra": ra, "dec": dec, "radius_deg": radius_deg, "mag_limit": mag_limit,
             "service": service}, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ network

    def _http(self, url: str, query: str) -> str:
        data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                       "QUERY": query}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/sync", data=data,
                                     headers={"User-Agent": "varstar-pipeline"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8")

    def cone(self, ra: float, dec: float, radius_deg: float, mag_limit: float) -> CatalogStars:
        cached = self._cached(ra, dec, radius_deg, mag_limit)
        if cached is not None:
            return cached
        query = GAIA_QUERY.format(ra=ra, dec=dec, radius=radius_deg, mag=mag_limit)
        problems = []
        for url in self.tap_urls:
            host = urllib.parse.urlparse(url).netloc or url
            try:
                text = self._fetch(url, query)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                problems.append(f"{host} : {getattr(exc, 'reason', exc)}")
                continue
            # Some services answer 200 with an HTML page (maintenance): check the CSV header.
            if not text.lstrip().lower().startswith("source_id"):
                problems.append(f"{host} : réponse inattendue (pas un CSV Gaia)")
                continue
            self._store(text, ra, dec, radius_deg, mag_limit, host)
            log.info("Catalogue Gaia téléchargé depuis %s", host)
            return _filter(parse_csv(text), ra, dec, radius_deg, mag_limit, f"Gaia DR3 ({host})")
        raise PipelineError("catalogue Gaia inaccessible — " + " ; ".join(problems)
                            + ". Réessayer plus tard, ou fournir un fichier CSV (astrometry.catalog).")
