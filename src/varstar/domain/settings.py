"""Typed settings (pure): one frozen dataclass per configuration section.

The YAML files are read by ``varstar.adapters.config_loader``. ``Settings.from_dict`` checks the
merged configuration (unknown keys, wrong types) and explains every problem in French.
Sections of milestones not built yet (quality, photometry…) stay as plain dicts in ``raw``.
``raw`` is also what is written to config.used.yaml and provenance.json, and what the cache
fingerprints hash: it is never modified.
"""

from __future__ import annotations

import copy
import datetime as dt
import types
import typing
from dataclasses import dataclass, fields
from typing import Any, Mapping


class SettingsError(ValueError):
    """Invalid configuration (French message naming the key)."""


@dataclass(frozen=True)
class PathsSettings:
    workdir: str | None
    library: str
    siril_cli: str
    astap_cli: str | None
    pixinsight_masters: str | None          # "auto" = <source>/output/master ; None = disabled


@dataclass(frozen=True)
class SirilSettings:
    python_init_guard: str                  # auto | on | off


@dataclass(frozen=True)
class InventorySettings:
    ignore_dirs: tuple[str, ...]
    include_removed: bool
    extensions: tuple[str, ...]
    session_day_offset_hours: float
    set_gap_hours: float
    read_pixels_for_units: bool


@dataclass(frozen=True)
class MatchingSettings:
    exptime_tolerance_s: float
    darkflat_exptime_tolerance_s: float
    temp_tolerance_c: float
    max_flat_age_days: float
    flat_same_session_hours: float
    trust_session_folder_flats: bool
    same_optical_setup: tuple[tuple[dt.date, dt.date], ...]   # inclusive periods
    allow_dark_scaling: bool
    dark_same_night: bool


@dataclass(frozen=True)
class MastersSettings:
    rejection_auto: bool
    sigma_low: float
    sigma_high: float
    percentile_low: float
    percentile_high: float
    flat_level_min_frac: float
    flat_level_max_frac: float


@dataclass(frozen=True)
class MasksSettings:
    hot_pixel_sigma: float
    hot_pixel_severe_sigma: float
    cold_pixel_sigma: float
    unstable_sigma_factor: float
    dead_pixel_flat_frac: float
    flat_local_block_px: int
    amp_glow_detect: bool
    amp_glow_block_px: int
    amp_glow_sigma: float
    amp_glow_min_adu: float
    bad_pixel_flags: tuple[str, ...]


@dataclass(frozen=True)
class QASettings:
    ccd_temp_delta_c: float
    focpos_tolerance_steps: float
    pointing_max_field_frac: float
    cloud_cover_max: float
    dew_margin_c: float
    min_darks: int
    min_flats: int
    pixel_step: int
    offset_jump_adu: float
    leak_frame_excess_adu: float
    flat_compare: bool
    flat_compare_max_frames: int
    flat_compare_bin: int
    flat_dust_pct: float
    flat_shape_pct: float
    dark_above_light_adu: float
    calibrated_background_min_adu: float


@dataclass(frozen=True)
class AstrometrySettings:
    engine: str
    catalog: str
    catalog_mag_limit: float
    tap_urls: tuple[str, ...]
    pixel_scale_arcsec: float | None
    pointing_tolerance_deg: float
    scale_tolerance: float
    rotation_step_deg: float
    detect_sigma: float
    max_stars: int
    match_radius_px: float
    min_matches: int
    max_rms_arcsec: float
    sip_order: int
    segment_angle_deg: float
    camera_rotation_tol_deg: float
    whole_train_rotation: bool | None          # default answer to the A5 question (None: ask)
    whole_train_rotation_nights: tuple[tuple[dt.date, bool], ...]   # answers per night (take precedence)
    scale_check_pct: float


@dataclass(frozen=True)
class RuntimeSettings:
    processes: int
    log_level: str


TYPED_SECTIONS = {"paths": PathsSettings, "siril": SirilSettings, "inventory": InventorySettings,
                  "matching": MatchingSettings, "masters": MastersSettings, "masks": MasksSettings,
                  "qa": QASettings, "astrometry": AstrometrySettings, "runtime": RuntimeSettings}


@dataclass(frozen=True, eq=False)
class Settings:
    paths: PathsSettings
    siril: SirilSettings
    inventory: InventorySettings
    matching: MatchingSettings
    masters: MastersSettings
    masks: MasksSettings
    qa: QASettings
    astrometry: AstrometrySettings
    runtime: RuntimeSettings
    raw: Mapping[str, Any]                  # the merged configuration, as read (never modified)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Settings":
        raw = copy.deepcopy(dict(data))
        missing = [s for s in TYPED_SECTIONS if not isinstance(raw.get(s), Mapping)]
        if missing:
            raise SettingsError(f"Configuration invalide : section(s) absente(s) {', '.join(missing)}")
        sections = {name: _section(name, sc, raw[name]) for name, sc in TYPED_SECTIONS.items()}
        return cls(**sections, raw=raw)

    def section(self, name: str) -> Any:
        """Raw content of a section (fingerprints, sections not typed yet)."""
        return copy.deepcopy(self.raw.get(name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Settings) and self.raw == other.raw

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- conversion


def _section(name: str, cls: type, data: Mapping[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise SettingsError(f"Configuration invalide : clé(s) inconnue(s) dans « {name} » : {', '.join(unknown)}")
    hints = typing.get_type_hints(cls)
    values = {}
    for key in known:
        if key not in data:
            raise SettingsError(f"Configuration invalide : {name}.{key} absent")
        if name == "matching" and key == "same_optical_setup":
            values[key] = tuple(optical_setup_periods(data[key]))
        elif name == "astrometry" and key == "whole_train_rotation_nights":
            values[key] = _night_answers(f"{name}.{key}", data[key])
        elif name == "astrometry" and key == "engine":
            values[key] = _engine(data[key])
        elif name == "siril" and key == "python_init_guard":
            values[key] = _guard_mode(data[key])
        else:
            values[key] = _convert(f"{name}.{key}", data[key], hints[key])
    return cls(**values)


def _convert(where: str, value: Any, hint: Any) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None:
            return None
        return _convert(where, value, args[0])
    if origin is tuple:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise SettingsError(f"Configuration invalide : {where} = {value!r} (liste attendue)")
        item = typing.get_args(hint)[0]
        return tuple(_convert(f"{where}[{i}]", v, item) for i, v in enumerate(value))
    if hint is bool:
        if not isinstance(value, bool):
            raise SettingsError(f"Configuration invalide : {where} = {value!r} (true ou false attendu)")
        return value
    if hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"Configuration invalide : {where} = {value!r} (nombre attendu)")
        if hint is int and float(value) != int(value):
            raise SettingsError(f"Configuration invalide : {where} = {value!r} (nombre entier attendu)")
        return hint(value)
    if hint is str:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise SettingsError(f"Configuration invalide : {where} = {value!r} (texte attendu)")
        return str(value)
    raise TypeError(f"type de réglage non pris en charge : {hint}")   # pragma: no cover


ENGINES = ("internal",)


def _engine(value: Any) -> str:
    if value not in ENGINES:
        raise SettingsError(f"Configuration invalide : astrometry.engine = {value!r} (moteurs disponibles : "
                            f"{', '.join(ENGINES)} ; Siril et ASTAP ne sont pas branchés, voir DECISIONS D17)")
    return str(value)


def _night_answers(where: str, raw: Any) -> tuple[tuple[dt.date, bool], ...]:
    """``{2021-09-02: true, …}`` → ((date, answer), …), sorted by date."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise SettingsError(f"Configuration invalide : {where} = {raw!r} (attendu {{AAAA-MM-JJ: true/false}})")
    out = []
    for night, answer in raw.items():
        try:
            d = _as_date(night)
        except ValueError:
            raise SettingsError(f"Configuration invalide : {where} : date invalide {night!r}") from None
        if not isinstance(answer, bool):
            raise SettingsError(f"Configuration invalide : {where}.{night} = {answer!r} (true ou false attendu)")
        out.append((d, answer))
    return tuple(sorted(out))


def _guard_mode(value: Any) -> str:
    if isinstance(value, bool):  # YAML 1.1 reads unquoted on/off as booleans
        return "on" if value else "off"
    return str(value)


def optical_setup_periods(raw: Any) -> list[tuple[dt.date, dt.date]]:
    """Parse matching.same_optical_setup into a list of inclusive (start, end) dates."""
    periods = []
    for item in raw or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SettingsError(f"same_optical_setup : période invalide {item!r} (attendu [début, fin])")
        try:
            start, end = (_as_date(x) for x in item)
        except ValueError as exc:
            raise SettingsError(f"same_optical_setup : date invalide dans {item!r} ({exc})") from None
        if end < start:
            start, end = end, start
        periods.append((start, end))
    return periods


def _as_date(x: Any) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))


def replace(settings: Settings, **sections: Mapping[str, Any]) -> Settings:
    """A copy with some keys changed, e.g. ``replace(cfg, qa={"pixel_step": 1})`` (checked again)."""
    raw = copy.deepcopy(dict(settings.raw))
    for name, values in sections.items():
        raw[name] = {**(raw.get(name) or {}), **values}
    return Settings.from_dict(raw)


__all__ = [c.__name__ for c in TYPED_SECTIONS.values()] + ["Settings", "SettingsError", "optical_setup_periods",
                                                            "replace"]
