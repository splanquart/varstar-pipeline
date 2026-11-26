"""Use case « masters » : plan, build (or reuse) the masters of each light group, masks, publish.

Strategy B (SPEC §4): dark of the same exposure/gain/binning/temperature, flats minus their dark
of flat, no bias; dark scaling only as an explicit, logged level B fallback.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.calibration import (MasterRecord, choose_rejection, compute_master_id, crosscheck_stats,
                                  dark_consistency, dark_label, dark_mode, dark_params, darkflat_kind, flat_label,
                                  flat_params, frames_of, group_key, mask_ids, master_summary, offset_dark_sets)
from ..domain.masks import BIT_NAMES, build_amp_glow, build_bpm
from ..domain.model import Frame, Inventory, LightGroup
from ..domain.plan import DarkflatRef, DarkRef, FlatRef, GroupPlan, MastersPlan
from ..domain.qa.darks import dark_candidates, effective_dark_sets
from ..domain.qa.model import report_items
from ..domain.qa.raw import saturation_adu
from ..domain.settings import Settings
from .check import load_report
from .ports import (ArtifactStore, FrameSource, MasterBuilder, MasterRepository, MastersReport, PixelStats,
                    Progress, ProvenanceRecorder, ReferenceMasters)

log = logging.getLogger("varstar")

PLAN_JSON = "masters/plan.json"
MASTERS_MD = "masters/masters.md"


def _silent(_: str) -> None:
    pass


@dataclass
class MasterFactory:
    """Reuses a master of the library when its inputs are identical, builds it otherwise."""

    cfg: Settings
    source: FrameSource
    builder: MasterBuilder
    repo: MasterRepository
    sources_root: Path

    def _id(self, kind: str, frames: list[Frame], params: dict[str, Any]) -> str:
        return compute_master_id(kind, [self.source.file_stat(f.path) for f in frames], params)

    def dark(self, frames: list[Frame], kind: str) -> MasterRecord:
        rej = choose_rejection(len(frames), self.cfg.masters)
        mid = self._id(kind, frames, dark_params(rej))
        found = self.repo.find(kind, mid)
        if found:
            log.info("Master %s %s déjà en bibliothèque (%s)", kind, mid, found.dir.name)
            return found
        label = dark_label(kind, frames)
        log.info("Construction du master %s %s (%d images, %s)", kind, label, len(frames), rej.name)
        return self.builder.build_dark(frames, kind, rej, mid, label, self.sources_root)

    def flat(self, frames: list[Frame], darkflat: MasterRecord) -> MasterRecord:
        rej = choose_rejection(len(frames), self.cfg.masters)
        mid = self._id("flat", frames, flat_params(rej, darkflat.id))
        found = self.repo.find("flat", mid)
        if found:
            log.info("Master flat %s déjà en bibliothèque (%s)", mid, found.dir.name)
            return found
        label = flat_label(frames)
        log.info("Construction du master flat %s (%d images, %s)", label, len(frames), rej.name)
        return self.builder.build_flat(frames, darkflat, rej, mid, label, self.sources_root)


def _masks(g: GroupPlan, dark_rec: MasterRecord | None, flat_rec: MasterRecord | None, inv: Inventory,
           cfg: Settings, masks_hash: str, store: ArtifactStore, repo: MasterRepository) -> None:
    """Bad pixel map of the dark/flat pair and amp glow mask of the dark (built once, in masks/)."""
    bpm_name, ag_name = mask_ids(dark_rec.id if dark_rec else None, flat_rec.id if flat_rec else None, masks_hash)
    bpm_rel = f"masks/{bpm_name}"
    dark_adu = repo.load_adu(dark_rec) if dark_rec is not None else None
    if not store.exists(bpm_rel):
        res = build_bpm(dark_adu, dark_rec.meta.get("sigma_single_frame_adu") if dark_rec else None,
                        repo.load_unstable(dark_rec) if dark_rec is not None else None,
                        repo.load_adu(flat_rec) if flat_rec is not None else None, cfg.masks,
                        saturation_adu=int(saturation_adu(inv)))
        header = {"VSDARK": dark_rec.id if dark_rec else "NONE", "VSFLAT": flat_rec.id if flat_rec else "NONE",
                  **{f"VSBIT{bit}": name for bit, name in BIT_NAMES.items()}}
        store.write_mask(bpm_rel, res.mask, header)
        store.write_json(bpm_rel[:-5] + ".json", {"counts": res.counts, "info": res.info})
    g.bpm = str(store.root / bpm_rel)
    if dark_rec is not None and cfg.masks.amp_glow_detect:
        ag_rel = f"masks/{ag_name}"
        if not store.exists(ag_rel):
            res = build_amp_glow(dark_adu, cfg.masks)
            store.write_mask(ag_rel, res.mask)
            store.write_json(ag_rel[:-5] + ".json", {"counts": res.counts, "info": res.info})
        g.ampglow = str(store.root / ag_rel)


def plan_group(g: LightGroup, inv: Inventory, cfg: Settings, factory: MasterFactory, pixels: PixelStats,
               masters: dict[str, MasterRecord], flat_labels: dict[str, list[str]], warnings: list[str],
               progress: Progress) -> tuple[GroupPlan, MasterRecord | None, MasterRecord | None]:
    """Masters of one light group (built or reused); masks are added by the caller."""
    key = group_key(g)
    entry = GroupPlan(key, g.session.isoformat(), g.label, g.filter, g.exptime, g.level, list(g.reasons),
                      [str(f.path) for f in g.frames])
    dark_rec = None
    dark_sets = list(g.dark.sets) if g.dark else []
    if dark_sets:
        # Never combine darks of different levels, never borrow an anomalous night (D16, P1).
        pool = {s.id: s for s in dark_candidates(inv, g, cfg.matching) + dark_sets}
        stats = pixels.frame_stats(frames_of(pool.values()), saturation_adu(inv), progress="niveau des darks")
        dark_sets, notes = effective_dark_sets(inv, g, stats, cfg)
        for note in notes:
            entry.reasons.append(note)
            warnings.append(f"{key} : {note}")
    if dark_sets:
        dark_rec = factory.dark(frames_of(dark_sets), "dark")
        masters[dark_rec.id] = dark_rec
        mode = dark_mode(dark_sets, g.exptime, cfg.matching.exptime_tolerance_s)
        entry.dark = DarkRef(dark_rec.id, str(dark_rec.path), mode, dark_rec.meta["n_frames"],
                             dark_rec.meta["exptime"])
        if mode == "scaled":
            kind, sets = offset_dark_sets(inv, g)
            off_rec = factory.dark(frames_of(sets), kind)
            entry.dark.offset_adu, entry.dark.offset_master = float(off_rec.meta["median_adu"]), off_rec.id
            warnings.append(f"{key} : dark mis à l'échelle depuis {dark_rec.meta['exptime']:g} s "
                            f"(offset estimé {entry.dark.offset_adu:.1f} ADU) — niveau B")
    flat_rec = None
    if g.flat and g.flat.level != "C" and g.flat.sets and g.darkflat and g.darkflat.sets:
        df_rec = factory.dark(frames_of(g.darkflat.sets), darkflat_kind(g.darkflat.sets))
        masters[df_rec.id] = df_rec
        flat_rec = factory.flat(frames_of(g.flat.sets), df_rec)
        masters[flat_rec.id] = flat_rec
        flat_labels[flat_rec.id] = g.flat.sets[0].folder_labels
        entry.darkflat = DarkflatRef(df_rec.id, str(df_rec.path), df_rec.meta["exptime"])
        entry.flat = FlatRef(flat_rec.id, str(flat_rec.path), flat_rec.meta["n_frames"])
        if flat_rec.meta.get("level_warnings"):
            warnings.append(f"{flat_rec.meta['label']} : {len(flat_rec.meta['level_warnings'])} flat(s) "
                            "hors de la plage 15–85 % de la dynamique")
    return entry, dark_rec, flat_rec


def run_masters(inv: Inventory, groups: list[LightGroup], cfg: Settings, *, source: FrameSource,
                pixels: PixelStats, builder: MasterBuilder, repo: MasterRepository,
                references: ReferenceMasters | None, store: ArtifactStore, provenance: ProvenanceRecorder,
                report: MastersReport, progress: Progress = _silent, now: dt.datetime | None = None) -> MastersPlan:
    """Step « masters »: publishes masters/plan.json and masters/masters.md, records the provenance."""
    factory = MasterFactory(cfg, source, builder, repo, inv.root)
    masters: dict[str, MasterRecord] = {}
    flat_labels: dict[str, list[str]] = {}
    plan = MastersPlan((now or dt.datetime.now()).isoformat(timespec="seconds"), str(repo.root), builder.version())
    masks_hash = provenance.config_fingerprint(cfg, ["masks"])[:8]

    for g in groups:
        progress(f"  masters pour {g.label or g.session} / {g.filter} …")
        entry, dark_rec, flat_rec = plan_group(g, inv, cfg, factory, pixels, masters, flat_labels, plan.warnings,
                                               progress)
        if dark_rec is not None or flat_rec is not None:
            _masks(entry, dark_rec, flat_rec, inv, cfg, masks_hash, store, repo)
        plan.groups[entry.key] = entry

    # Cross-validation with PixInsight masters + Q2 dark consistency.
    for rec in masters.values():
        info = master_summary(rec)
        cc = _crosscheck_with(rec, references, flat_labels.get(rec.id, []), repo)
        if cc:
            info["pixinsight"] = cc
        if rec.kind == "dark":
            info["consistency"] = dark_consistency(rec.meta.get("frame_stats", []))
        plan.masters[rec.id] = info

    data = plan.to_dict()
    store.write_json(PLAN_JSON, data)
    qa_report = load_report(store)
    qa_items = report_items(qa_report, ("header", "raw")) if qa_report else None
    store.write_text(MASTERS_MD, report.markdown(data, qa_items, mask_summaries(plan, store)))
    provenance.record("masters", cfg, {
        "siril_version": builder.version(), "siril_python_init_guard": builder.status(),
        "library": str(repo.root),
        "masters": {mid: {"kind": m.kind, "dir": str(m.dir), "sources": m.meta["sources"],
                          "rejection": m.meta["rejection"], "n": m.meta["n_frames"]}
                    for mid, m in masters.items()},
    })
    for w in plan.warnings:
        log.warning(w)
    return plan


def _crosscheck_with(rec: MasterRecord, references: ReferenceMasters | None, folder_labels: list[str],
                     repo: MasterRepository) -> dict[str, Any] | None:
    if references is None:
        return None
    path = references.find(rec, folder_labels)
    if path is None:
        return None
    name = Path(path).name
    try:
        theirs = references.load_adu(path)
    except (KeyError, ValueError) as exc:
        return {"pixinsight": name, "error": str(exc)}
    return {"pixinsight": name, **crosscheck_stats(rec.kind, repo.load_adu(rec), theirs)}


def mask_summaries(plan: MastersPlan, store: ArtifactStore) -> dict[str, dict[str, Any]]:
    """Counts/info of every mask of the plan (read back from masks/*.json)."""
    out: dict[str, dict[str, Any]] = {}
    for g in plan.groups.values():
        for path in (g.bpm, g.ampglow):
            if path and path not in out:
                rel = str(Path(path).relative_to(store.root))[:-5] + ".json"
                data = store.read_json(rel)
                if data is not None:
                    out[path] = data
    return out


def load_plan(store: ArtifactStore) -> MastersPlan | None:
    data = store.read_json(PLAN_JSON)
    return None if data is None else MastersPlan.from_dict(data)

