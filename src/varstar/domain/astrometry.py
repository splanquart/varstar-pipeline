"""Astrometry (M3), pure: tangent projection, WCS solutions, plate solving by pattern voting, segments.

The solver is a *near* solver: it knows the approximate pointing (RA/DEC of the mount in the
header) and the pixel scale (FOCALLEN, XPIXSZ). It searches the rotation and the mirror
(parity), votes for the translation that superimposes the brightest detected stars on the
brightest catalogue stars, then refines an affine fit by iterative nearest-neighbour
matching, and finally an optional SIP distortion polynomial.

Pixel coordinates are 0-based numpy coordinates (x = column, y = row); FITS keywords
(CRPIX) are 1-based. Intermediate world coordinates of RA---TAN are the standard
coordinates (xi, eta) in degrees. Nothing here reads a file or resamples an image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .settings import AstrometrySettings
from .stars import Stars, detect_stars

ASTROMETRY_VERSION = 2   # 2: vote reaches past the saturated stars (sept23 OIII)


# =========================================================================== projection


def radec_to_std(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Gnomonic (TAN) projection: (ra, dec) → standard coordinates (xi, eta), all in degrees.

    Points more than 90° from the tangent point get NaN.
    """
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    a0, d0 = math.radians(ra0), math.radians(dec0)
    da = ra - a0
    cosc = math.sin(d0) * np.sin(dec) + math.cos(d0) * np.cos(dec) * np.cos(da)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(dec) * np.sin(da) / cosc
        eta = (math.cos(d0) * np.sin(dec) - math.sin(d0) * np.cos(dec) * np.cos(da)) / cosc
    bad = cosc <= 0
    xi, eta = np.degrees(xi), np.degrees(eta)
    return np.where(bad, np.nan, xi), np.where(bad, np.nan, eta)


def std_to_radec(xi: np.ndarray, eta: np.ndarray, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse gnomonic projection (degrees in, degrees out, RA in [0, 360))."""
    x, y = np.radians(np.asarray(xi, float)), np.radians(np.asarray(eta, float))
    a0, d0 = math.radians(ra0), math.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rho > 0, y * np.sin(c) / rho, 0.0)
    dec = np.arcsin(np.clip(np.cos(c) * math.sin(d0) + ratio * math.cos(d0), -1, 1))
    ra = a0 + np.arctan2(x * np.sin(c), rho * math.cos(d0) * np.cos(c) - y * math.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle distance (haversine), arcseconds."""
    r1, d1, r2, d2 = (np.radians(np.asarray(v, float)) for v in (ra1, dec1, ra2, dec2))
    s = np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600.0


def expected_scale_arcsec(focallen_mm: float | None, xpixsz_um: float | None) -> float | None:
    """Pixel scale from the header. N.I.N.A. writes XPIXSZ already multiplied by the binning."""
    if not focallen_mm or not xpixsz_um:
        return None
    return 206.264806 * float(xpixsz_um) / float(focallen_mm)


# =========================================================================== catalogue, hints, solutions


@dataclass(frozen=True)
class CatalogStars:
    """Reference stars (ICRS, degrees), brightest first after ``sorted_by_mag``."""

    ra: np.ndarray
    dec: np.ndarray
    mag: np.ndarray
    source: str = ""           # e.g. "Gaia DR3 (G < 13)", or a file name
    ids: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.ra.size)

    def sorted_by_mag(self) -> "CatalogStars":
        o = np.argsort(self.mag, kind="stable")
        ids = tuple(self.ids[i] for i in o) if self.ids else ()
        return CatalogStars(self.ra[o], self.dec[o], self.mag[o], self.source, ids)


@dataclass(frozen=True)
class SolveHint:
    """What the header says: approximate centre and pixel scale, image size."""

    ra: float
    dec: float
    scale_arcsec: float
    width: int
    height: int
    angle_deg: float | None = None      # rotation found on a previous image of the series
    parity: int | None = None
    angle_window_deg: float = 3.0

    def with_prior(self, result: "SolveResult") -> "SolveHint":
        """Same series: search only near the rotation, mirror and scale already found."""
        st = result.stats
        if not result.ok or result.solution is None or "vote_angle_deg" not in st:
            return self
        return SolveHint(self.ra, self.dec, result.solution.scale_arcsec, self.width, self.height,
                         float(st["vote_angle_deg"]), int(st["vote_parity"]), self.angle_window_deg)

    @property
    def field_deg(self) -> tuple[float, float]:
        return self.width * self.scale_arcsec / 3600, self.height * self.scale_arcsec / 3600

    @property
    def half_diagonal_deg(self) -> float:
        w, h = self.field_deg
        return 0.5 * math.hypot(w, h)


@dataclass(frozen=True)
class WcsSolution:
    """TAN (+ SIP) solution. ``cd`` in degrees per pixel; SIP terms as (p, q, coefficient)."""

    crval: tuple[float, float]
    crpix: tuple[float, float]                  # FITS, 1-based
    cd: tuple[tuple[float, float], tuple[float, float]]
    sip_order: int = 0
    sip_a: tuple[tuple[int, int, float], ...] = ()
    sip_b: tuple[tuple[int, int, float], ...] = ()

    # ---------------------------------------------------------------- geometry

    @property
    def cd_matrix(self) -> np.ndarray:
        return np.array(self.cd, dtype=float)

    @property
    def scale_arcsec(self) -> float:
        """Mean pixel scale (sqrt of |det CD|), arcsec per pixel."""
        return math.sqrt(abs(float(np.linalg.det(self.cd_matrix)))) * 3600

    @property
    def parity(self) -> int:
        """-1 for the usual sky orientation (north up → east left), +1 for a mirrored image."""
        return 1 if float(np.linalg.det(self.cd_matrix)) > 0 else -1

    @property
    def orientation_deg(self) -> float:
        """Direction of north on the pixel grid, degrees from +y towards +x, in [0, 360).

        A meridian flip changes it by 180°; a camera rotation by the rotation angle.
        """
        n = np.linalg.solve(self.cd_matrix, np.array([0.0, 1.0]))
        return math.degrees(math.atan2(n[0], n[1])) % 360.0

    @property
    def skew(self) -> float:
        """|1 − ratio of the two axis scales| (0 for a square, orthogonal pixel grid)."""
        sx = float(np.hypot(*self.cd_matrix[:, 0]))
        sy = float(np.hypot(*self.cd_matrix[:, 1]))
        return abs(1 - sx / sy) if sy else 1.0

    def _sip(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        du = np.zeros_like(u, dtype=float)
        dv = np.zeros_like(v, dtype=float)
        for p, q, c in self.sip_a:
            du += c * u ** p * v ** q
        for p, q, c in self.sip_b:
            dv += c * u ** p * v ** q
        return du, dv

    def pixel_to_world(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """0-based pixel → (ra, dec) degrees."""
        u = np.asarray(x, float) + 1 - self.crpix[0]
        v = np.asarray(y, float) + 1 - self.crpix[1]
        if self.sip_order:
            du, dv = self._sip(u, v)
            u, v = u + du, v + dv
        cd = self.cd_matrix
        xi = cd[0, 0] * u + cd[0, 1] * v
        eta = cd[1, 0] * u + cd[1, 1] * v
        return std_to_radec(xi, eta, *self.crval)

    def world_to_pixel(self, ra, dec) -> tuple[np.ndarray, np.ndarray]:
        """(ra, dec) degrees → 0-based pixel (SIP inverted by fixed-point iteration)."""
        xi, eta = radec_to_std(ra, dec, *self.crval)
        inv = np.linalg.inv(self.cd_matrix)
        U = inv[0, 0] * xi + inv[0, 1] * eta
        V = inv[1, 0] * xi + inv[1, 1] * eta
        u, v = U.copy(), V.copy()
        if self.sip_order:
            for _ in range(20):
                du, dv = self._sip(u, v)
                u, v = U - du, V - dv
        return u + self.crpix[0] - 1, v + self.crpix[1] - 1

    # ---------------------------------------------------------------- serialisation

    def header(self) -> dict[str, Any]:
        """FITS WCS keywords (readable by astropy.wcs, DS9, PixInsight, Siril)."""
        sip = self.sip_order >= 2 and (self.sip_a or self.sip_b)
        h: dict[str, Any] = {
            "WCSAXES": 2, "CTYPE1": "RA---TAN-SIP" if sip else "RA---TAN",
            "CTYPE2": "DEC--TAN-SIP" if sip else "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
            "CRVAL1": self.crval[0], "CRVAL2": self.crval[1], "CRPIX1": self.crpix[0], "CRPIX2": self.crpix[1],
            "CD1_1": self.cd[0][0], "CD1_2": self.cd[0][1], "CD2_1": self.cd[1][0], "CD2_2": self.cd[1][1],
            "RADESYS": "ICRS", "EQUINOX": 2000.0,
        }
        if sip:
            h["A_ORDER"] = h["B_ORDER"] = self.sip_order
            for p, q, c in self.sip_a:
                h[f"A_{p}_{q}"] = c
            for p, q, c in self.sip_b:
                h[f"B_{p}_{q}"] = c
        return h

    def to_dict(self) -> dict[str, Any]:
        return {"crval": list(self.crval), "crpix": list(self.crpix), "cd": [list(r) for r in self.cd],
                "sip_order": self.sip_order, "sip_a": [list(t) for t in self.sip_a],
                "sip_b": [list(t) for t in self.sip_b]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WcsSolution":
        return cls(tuple(d["crval"]), tuple(d["crpix"]), tuple(tuple(r) for r in d["cd"]),
                   int(d.get("sip_order", 0)), tuple((int(p), int(q), float(c)) for p, q, c in d.get("sip_a", [])),
                   tuple((int(p), int(q), float(c)) for p, q, c in d.get("sip_b", [])))


@dataclass(frozen=True)
class SolveSettings:
    """Solver tuning (see the « astrometry » section of the configuration)."""

    rotation_step_deg: float = 0.5
    scale_tolerance: float = 0.05       # accepted |scale / expected − 1|
    pointing_tolerance_deg: float = 0.5  # how far the true centre may be from the header RA/DEC
    vote_stars: int = 60
    vote_catalog: int = 200
    vote_bin_px: float = 8.0
    match_radius_px: float = 2.0
    min_matches: int = 20
    max_rms_arcsec: float = 1.5
    max_skew: float = 0.02
    sip_order: int = 3
    detect_sigma: float = 5.0
    max_stars: int = 300
    saturation_adu: float | None = None

    @classmethod
    def of(cls, a: AstrometrySettings, saturation_adu: float | None = None) -> "SolveSettings":
        return cls(rotation_step_deg=a.rotation_step_deg, scale_tolerance=a.scale_tolerance,
                   pointing_tolerance_deg=a.pointing_tolerance_deg, match_radius_px=a.match_radius_px,
                   min_matches=a.min_matches, max_rms_arcsec=a.max_rms_arcsec, sip_order=a.sip_order,
                   detect_sigma=a.detect_sigma, max_stars=a.max_stars, saturation_adu=saturation_adu)


@dataclass(frozen=True)
class SolveResult:
    ok: bool
    solution: WcsSolution | None = None
    reason: str | None = None           # French, when not ok
    n_stars: int = 0
    n_catalog: int = 0
    n_matched: int = 0
    rms_arcsec: float | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"ok": self.ok, "reason": self.reason, "n_stars": self.n_stars,
                             "n_catalog": self.n_catalog, "n_matched": self.n_matched,
                             "rms_arcsec": self.rms_arcsec, "stats": dict(self.stats)}
        if self.solution is not None:
            s = self.solution
            d["wcs"] = s.to_dict()
            d["header"] = s.header()
            d.update(scale_arcsec=s.scale_arcsec, orientation_deg=s.orientation_deg, parity=s.parity)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SolveResult":
        sol = WcsSolution.from_dict(d["wcs"]) if d.get("wcs") else None
        return cls(bool(d["ok"]), sol, d.get("reason"), int(d.get("n_stars", 0)), int(d.get("n_catalog", 0)),
                   int(d.get("n_matched", 0)), d.get("rms_arcsec"), dict(d.get("stats", {})))


def failed(reason: str, **kw: Any) -> SolveResult:
    return SolveResult(False, None, reason, **kw)


# =========================================================================== solving


def _sip_terms(order: int) -> list[tuple[int, int]]:
    return [(p, n - p) for n in range(2, order + 1) for p in range(n, -1, -1)]


def _vote(px: np.ndarray, py: np.ndarray, cu: np.ndarray, cv: np.ndarray, s: SolveSettings,
          max_offset_px: float, angles_deg: np.ndarray, parities: Sequence[int]) -> tuple[int, int, float, float, float]:
    """Best (count, parity, angle, tx, ty) superimposing star pattern (px, py) on (cu, cv)."""
    b = s.vote_bin_px
    nb = int(math.ceil(2 * max_offset_px / b)) + 2
    best = (0, 1, 0.0, 0.0, 0.0)
    angles = np.radians(angles_deg)
    for parity in parities:
        qx0 = parity * px
        for th in angles:
            c, sn = math.cos(th), math.sin(th)
            qx = c * qx0 - sn * py
            qy = sn * qx0 + c * py
            dx = (cu[None, :] - qx[:, None]).ravel()
            dy = (cv[None, :] - qy[:, None]).ravel()
            ix = np.floor((dx + max_offset_px) / b).astype(int)
            iy = np.floor((dy + max_offset_px) / b).astype(int)
            ok = (ix >= 0) & (ix < nb - 1) & (iy >= 0) & (iy < nb - 1)
            if not ok.any():
                continue
            hist = np.bincount(iy[ok] * nb + ix[ok], minlength=nb * nb).reshape(nb, nb)
            # 2×2 sums: a true offset on a bin edge is not split in two.
            h2 = hist[:-1, :-1] + hist[1:, :-1] + hist[:-1, 1:] + hist[1:, 1:]
            k = int(np.argmax(h2))
            count = int(h2.flat[k])
            if count > best[0]:
                jy, jx = divmod(k, nb - 1)
                tx = (jx + 1) * b - max_offset_px
                ty = (jy + 1) * b - max_offset_px
                best = (count, parity, float(th), tx, ty)
    return best


def _affine_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×2 matrix M with dst ≈ [x, y, 1] @ M (least squares)."""
    A = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M


def _match(pred: np.ndarray, tree: cKDTree, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one nearest-neighbour pairs (star index, catalogue index) closer than ``radius``."""
    dist, idx = tree.query(pred, distance_upper_bound=radius)
    ok = np.isfinite(dist)
    si, ci, dd = np.nonzero(ok)[0], idx[ok], dist[ok]
    order = np.argsort(dd)
    seen: set[int] = set()
    keep = []
    for k in order:
        c = int(ci[k])
        if c not in seen:
            seen.add(c)
            keep.append(k)
    keep = np.array(sorted(keep), dtype=int)
    return si[keep], ci[keep]


def _fit_wcs(x: np.ndarray, y: np.ndarray, ra: np.ndarray, dec: np.ndarray, crpix: tuple[float, float],
             crval: tuple[float, float], sip_order: int) -> WcsSolution:
    """TAN (+ SIP) fit with CRPIX fixed; CRVAL moved so that CRPIX projects on the tangent point.

    SIP is exactly CD × polynomial: (xi, eta) = CD·(u, v) + Σ CD·(a_pq, b_pq) u^p v^q. Fitting xi and
    eta as polynomials of (u, v) is therefore one linear least-squares problem; the linear part gives
    CD and CD⁻¹ times the higher-order coefficients gives the SIP terms.
    """
    u = x + 1 - crpix[0]
    v = y + 1 - crpix[1]
    terms = _sip_terms(sip_order) if sip_order >= 2 else []
    if len(x) < 3 * (len(terms) + 3) + 10:
        terms = []
    norm = float(max(np.abs(u).max(), np.abs(v).max(), 1.0))
    un, vn = u / norm, v / norm
    A = np.column_stack([np.ones_like(u), u, v] + [un ** p * vn ** q for p, q in terms])
    sol = None
    for _ in range(5):
        xi, eta = radec_to_std(ra, dec, *crval)
        cx, *_ = np.linalg.lstsq(A, xi, rcond=None)
        ce, *_ = np.linalg.lstsq(A, eta, rcond=None)
        cd = np.array([[cx[1], cx[2]], [ce[1], ce[2]]])
        sip_a: tuple[tuple[int, int, float], ...] = ()
        sip_b: tuple[tuple[int, int, float], ...] = ()
        if terms:
            inv = np.linalg.inv(cd)
            hi = inv @ np.vstack([cx[3:], ce[3:]])
            sip_a = tuple((p, q, float(hi[0, k]) / norm ** (p + q)) for k, (p, q) in enumerate(terms))
            sip_b = tuple((p, q, float(hi[1, k]) / norm ** (p + q)) for k, (p, q) in enumerate(terms))
        new_ra, new_dec = std_to_radec(cx[0], ce[0], *crval)
        done = abs(cx[0]) < 1e-10 and abs(ce[0]) < 1e-10
        crval = (float(new_ra), float(new_dec))
        sol = WcsSolution(crval, crpix, ((float(cd[0, 0]), float(cd[0, 1])), (float(cd[1, 0]), float(cd[1, 1]))),
                          sip_order if terms else 0, sip_a, sip_b)
        if done:
            break
    assert sol is not None
    return sol


def solve_field(stars: Stars, catalog: CatalogStars, hint: SolveHint, s: SolveSettings = SolveSettings()) -> SolveResult:
    """Plate-solve one image from its detected stars. Failures are returned with a French reason."""
    base = {"n_stars": stars.n}
    min_vote = max(5, min(s.min_matches // 2, 12))
    if stars.n < min_vote:
        return failed(f"{stars.n} étoile(s) détectée(s) seulement (nuages, buée, mise au point ?)", **base)
    cat = catalog.sorted_by_mag()
    xi, eta = radec_to_std(cat.ra, cat.dec, hint.ra, hint.dec)
    scale = hint.scale_arcsec
    cu, cv = xi * 3600 / scale, eta * 3600 / scale
    reach = (hint.half_diagonal_deg + s.pointing_tolerance_deg) * 3600 / scale
    inside = np.isfinite(cu) & (np.hypot(cu, cv) <= reach)
    cu, cv = cu[inside], cv[inside]
    cra, cdec = cat.ra[inside], cat.dec[inside]
    base["n_catalog"] = int(cu.size)
    if cu.size < min_vote:
        return failed(f"{cu.size} étoile(s) du catalogue dans le champ : catalogue trop pauvre ou mauvais pointage",
                      **base)
    cx, cy = (hint.width - 1) / 2, (hint.height - 1) / 2
    px, py = stars.x - cx, stars.y - cy

    # 1. Coarse: rotation, parity and translation by voting (brightest stars only).
    # Saturated stars are rejected by the detection: they are the brightest catalogue stars, so the
    # catalogue list grows by as many (OIII under a bright sky: dozens of saturated stars).
    nv, nc = min(s.vote_stars, stars.n), min(s.vote_catalog + stars.n_saturated, cu.size)
    max_off = s.pointing_tolerance_deg * 3600 / scale + s.vote_bin_px
    full_search = (np.arange(0.0, 360.0, s.rotation_step_deg), (1, -1), (1.0, 1 - s.scale_tolerance / 2,
                                                                         1 + s.scale_tolerance / 2))
    searches = [full_search]
    if hint.angle_deg is not None and hint.parity in (1, -1):
        w = hint.angle_window_deg
        narrow = np.arange(hint.angle_deg - w, hint.angle_deg + w + 1e-9, s.rotation_step_deg) % 360.0
        searches.insert(0, (narrow, (hint.parity,), (1.0,)))
    best = None
    for angles, parities, factors in searches:
        for f in factors:
            cand = _vote(px[:nv] * f, py[:nv] * f, cu[:nc], cv[:nc], s, max_off, angles, parities)
            if best is None or cand[0] > best[0][0]:
                best = (cand, f)
        if best is not None and best[0][0] >= max(min_vote, nv // 3):
            break   # the narrow search (same series) found it: no need for the full one
    assert best is not None
    (count, parity, th, tx, ty), f = best
    if count < min_vote:
        return failed(f"aucune correspondance entre les étoiles et le catalogue (meilleur vote : {count} étoiles) — "
                      "pointage ou échelle de l'en-tête faux ?", **base)

    # 2. Refine: affine fit, nearest-neighbour matching with a shrinking radius (all stars).
    c, sn = math.cos(th), math.sin(th)
    M = np.array([[f * c * parity, f * sn * parity], [-f * sn, f * c], [tx, ty]])
    src = np.column_stack([px, py])
    tree = cKDTree(np.column_stack([cu, cv]))
    si = ci = np.empty(0, dtype=int)
    for radius in (3 * s.vote_bin_px, 2 * s.vote_bin_px, s.vote_bin_px, 3 * s.match_radius_px, s.match_radius_px):
        pred = np.column_stack([src, np.ones(len(src))]) @ M
        si, ci = _match(pred, tree, radius)
        if si.size < 3:
            break
        M = _affine_fit(src[si], np.column_stack([cu[ci], cv[ci]]))
    n_match = int(si.size)
    base["n_matched"] = n_match
    if n_match < s.min_matches:
        return failed(f"{n_match} étoile(s) appariée(s) seulement (minimum {s.min_matches})", **base)

    # 3. WCS fit on the sky, then one more matching pass with the final solution.
    crpix = (cx + 1, cy + 1)
    sol = _fit_wcs(stars.x[si], stars.y[si], cra[ci], cdec[ci], crpix, (hint.ra, hint.dec), s.sip_order)
    full = CatalogStars(cra, cdec, cat.mag[inside])
    pxs, pys = sol.world_to_pixel(full.ra, full.dec)
    tree2 = cKDTree(np.column_stack([pxs, pys]))
    si, ci = _match(np.column_stack([stars.x, stars.y]), tree2, s.match_radius_px)
    if si.size >= s.min_matches:
        sol = _fit_wcs(stars.x[si], stars.y[si], cra[ci], cdec[ci], crpix, sol.crval, s.sip_order)
    n_match = int(si.size)
    base["n_matched"] = n_match
    ra_fit, dec_fit = sol.pixel_to_world(stars.x[si], stars.y[si])
    res = angular_sep_arcsec(ra_fit, dec_fit, cra[ci], cdec[ci])
    rms = float(np.sqrt(np.mean(res ** 2))) if res.size else float("inf")
    stats = {"vote": count, "n_saturated": stars.n_saturated, "vote_angle_deg": math.degrees(th) % 360.0, "vote_parity": parity,
             "residual_p90_arcsec": float(np.percentile(res, 90)) if res.size else None}
    expected = hint.scale_arcsec
    ratio = sol.scale_arcsec / expected - 1
    common = {**base, "rms_arcsec": rms, "stats": stats}
    if n_match < s.min_matches:
        return SolveResult(False, sol, f"{n_match} étoile(s) appariée(s) seulement (minimum {s.min_matches})", **common)
    if rms > s.max_rms_arcsec:
        return SolveResult(False, sol, f"résidus trop forts : {rms:.2f}″ (maximum {s.max_rms_arcsec:g}″)", **common)
    if abs(ratio) > s.scale_tolerance:
        return SolveResult(False, sol, f"échelle {sol.scale_arcsec:.3f}″/px trop éloignée de l'échelle attendue "
                                       f"{expected:.3f}″/px ({100 * ratio:+.1f} %)", **common)
    if sol.skew > s.max_skew:
        return SolveResult(False, sol, f"solution déformée (axes inégaux de {100 * sol.skew:.1f} %)", **common)
    return SolveResult(True, sol, None, **common)


def solve_image(img: np.ndarray, catalog: CatalogStars, hint: SolveHint,
                s: SolveSettings = SolveSettings()) -> SolveResult:
    """Detect the stars of a calibrated image, then plate-solve it."""
    stars = detect_stars(img, threshold_sigma=s.detect_sigma, max_stars=s.max_stars, saturation=s.saturation_adu)
    res = solve_field(stars, catalog, hint, s)
    stats = {**res.stats, "background_adu": stars.background_adu, "noise_adu": stars.noise_adu}
    return SolveResult(res.ok, res.solution, res.reason, res.n_stars, res.n_catalog, res.n_matched,
                       res.rms_arcsec, stats)


@dataclass(frozen=True)
class SolveJob:
    """One calibrated image to solve: ``path`` identifies it (calibrated file), ``hint`` comes from the header."""

    path: str
    hint: SolveHint | None
    time: str | None = None            # DATE-OBS (ISO), orders a series
    problem: str | None = None         # why no hint could be built (French)


def solve_series(jobs: Sequence[SolveJob], load, catalog: CatalogStars,
                 s: SolveSettings = SolveSettings()) -> dict[str, SolveResult]:
    """Solve a time-ordered series; each solution guides the next image (narrow rotation search).

    ``load(path)`` returns the image (it may raise ``OSError``: the image fails, the series goes on).
    """
    out: dict[str, SolveResult] = {}
    prior: SolveResult | None = None
    for job in sorted(jobs, key=lambda j: (j.time or "", j.path)):
        if job.hint is None:
            out[job.path] = failed(job.problem or "indices de pointage absents")
            continue
        try:
            img = load(job.path)
        except OSError as exc:
            out[job.path] = failed(f"image illisible ({exc.__class__.__name__})")
            continue
        hint = job.hint.with_prior(prior) if prior is not None else job.hint
        res = solve_image(img, catalog, hint, s)
        out[job.path] = res
        if res.ok:
            prior = res
    return out


def catalog_cone(hints: Sequence[SolveHint], pointing_tolerance_deg: float) -> tuple[float, float, float]:
    """(ra, dec, radius) of one catalogue cone covering every image of the run, rounded up to 0.1°."""
    if not hints:
        raise ValueError("aucune image")
    ra = np.radians([h.ra for h in hints])
    dec = np.radians([h.dec for h in hints])
    vec = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).mean(axis=0)
    ra0 = math.degrees(math.atan2(vec[1], vec[0])) % 360.0
    dec0 = math.degrees(math.atan2(vec[2], math.hypot(vec[0], vec[1])))
    spread = float(np.max(angular_sep_arcsec([h.ra for h in hints], [h.dec for h in hints], ra0, dec0))) / 3600
    radius = spread + max(h.half_diagonal_deg for h in hints) + pointing_tolerance_deg
    return round(ra0, 3), round(dec0, 3), math.ceil(radius * 10) / 10


# =========================================================================== segments


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, degrees in [0, 180]."""
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def assign_segments(items: Sequence[tuple[Any, float]], tolerance_deg: float = 20.0) -> dict[Any, int]:
    """Segments of a night: time-ordered (key, orientation) → segment number (1, 2, …).

    A new segment starts when the orientation differs by more than ``tolerance_deg`` from the
    current segment's (e.g. 180° after a meridian flip). Returning to an earlier orientation
    starts a new segment too: segments are contiguous in time.
    """
    out: dict[Any, int] = {}
    current: float | None = None
    seg = 0
    for key, angle in items:
        if current is None or angle_diff_deg(angle, current) > tolerance_deg:
            seg += 1
            current = angle
        out[key] = seg
    return out


def camera_angle_deg(orientation_deg: float) -> float:
    """Orientation modulo 180°: the camera rotation, independent of a meridian flip."""
    return orientation_deg % 180.0


def circular_median_deg(angles: Iterable[float], period: float = 360.0) -> float:
    """Median of angles around their circular mean (period 360°, or 180° for camera angles)."""
    a = np.asarray(list(angles), float)
    if a.size == 0:
        return float("nan")
    k = 2 * math.pi / period
    mean = math.atan2(float(np.mean(np.sin(a * k))), float(np.mean(np.cos(a * k)))) / k
    d = (a - mean + period / 2) % period - period / 2
    return float((mean + np.median(d)) % period)
