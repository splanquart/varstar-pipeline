"""The masters plan (masters/plan.json): which masters and masks calibrate each light group.

Typed in memory, plain JSON on disk (``to_dict`` / ``from_dict`` keep the historical layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class DarkRef:
    id: str
    path: str
    mode: str                      # "direct" | "scaled" (level B fallback)
    n: int
    exptime: float
    offset_adu: float | None = None    # scaled darks only: uniform offset removed before scaling
    offset_master: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "path": self.path, "mode": self.mode, "n": self.n, "exptime": self.exptime}
        if self.mode == "scaled":
            d.update(offset_adu=self.offset_adu, offset_master=self.offset_master)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DarkRef":
        return cls(d["id"], d["path"], d["mode"], d["n"], d["exptime"], d.get("offset_adu"), d.get("offset_master"))


@dataclass
class FlatRef:
    id: str
    path: str
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "path": self.path, "n": self.n}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FlatRef":
        return cls(d["id"], d["path"], d["n"])


@dataclass
class DarkflatRef:
    id: str
    path: str
    exptime: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "path": self.path, "exptime": self.exptime}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DarkflatRef":
        return cls(d["id"], d["path"], d["exptime"])


def _opt(cls, d):
    return None if d is None else cls.from_dict(d)


@dataclass
class GroupPlan:
    key: str
    session: str                   # ISO date of the night
    label: str | None
    filter: str
    exptime: float
    level: str
    reasons: list[str]
    files: list[str]               # raw lights (absolute paths)
    dark: DarkRef | None = None
    flat: FlatRef | None = None
    darkflat: DarkflatRef | None = None
    bpm: str | None = None         # mask paths (absolute)
    ampglow: str | None = None

    @property
    def display(self) -> str:
        return f"{self.label or self.session} / {self.filter}"

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "session": self.session, "label": self.label, "filter": self.filter,
                "exptime": self.exptime, "level": self.level, "reasons": list(self.reasons),
                "files": list(self.files), "dark": self.dark.to_dict() if self.dark else None,
                "flat": self.flat.to_dict() if self.flat else None,
                "darkflat": self.darkflat.to_dict() if self.darkflat else None, "bpm": self.bpm,
                "ampglow": self.ampglow}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GroupPlan":
        return cls(d["key"], d["session"], d.get("label"), d["filter"], d["exptime"], d["level"],
                   list(d.get("reasons", [])), list(d["files"]), _opt(DarkRef, d.get("dark")),
                   _opt(FlatRef, d.get("flat")), _opt(DarkflatRef, d.get("darkflat")), d.get("bpm"),
                   d.get("ampglow"))


@dataclass
class MastersPlan:
    created: str
    library: str
    siril_version: str
    groups: dict[str, GroupPlan] = field(default_factory=dict)
    masters: dict[str, dict[str, Any]] = field(default_factory=dict)   # id → summary (+ cross-checks)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"created": self.created, "library": self.library, "siril_version": self.siril_version,
                "groups": {k: g.to_dict() for k, g in self.groups.items()}, "masters": self.masters,
                "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MastersPlan":
        return cls(d["created"], d["library"], d["siril_version"],
                   {k: GroupPlan.from_dict(g) for k, g in d.get("groups", {}).items()},
                   dict(d.get("masters", {})), list(d.get("warnings", [])))
