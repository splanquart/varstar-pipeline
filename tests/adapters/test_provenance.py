"""Provenance adapter: fingerprints, manifests and provenance.json."""

import json

from varstar.adapters.provenance import (PACKAGE_DIR, ProvenanceStore, code_files, code_fingerprint,
                                         software_versions)


def test_code_fingerprint_covers_every_layer():
    rel = {str(p.relative_to(PACKAGE_DIR)) for p in code_files()}
    assert {"domain/model.py", "application/pipeline.py", "adapters/siril.py", "cli/bootstrap.py",
            "defaults.yaml", "__init__.py"} <= rel
    assert code_fingerprint() == code_fingerprint()


def test_provenance_json_keeps_every_step(tmp_path, cfg):
    store = ProvenanceStore(tmp_path)
    store.record("masters", cfg, {"n": 1})
    store.record("calibrate", cfg, {"errors": {}})
    prov = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert set(prov["steps"]) == {"masters", "calibrate"} and prov["steps"]["masters"]["n"] == 1
    assert prov["software"]["varstar"] == software_versions()["varstar"] and prov["config"] == cfg.raw
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "inventory.json").write_text("{broken")
    assert store.read_manifest("inventory") is None
    (tmp_path / "provenance.json").write_text("{broken")
    store.record("inventory", cfg, {})
    assert list(json.loads((tmp_path / "provenance.json").read_text())["steps"]) == ["inventory"]
