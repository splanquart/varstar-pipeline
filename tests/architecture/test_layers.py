"""Architecture rules, checked statically on the imports of src/varstar (docs: plan hexagonal).

Layers (sub-packages of ``varstar``), each may only depend on the layers to its left:

    domain  ←  application  ←  adapters / cli

- ``domain``: pure business rules. Standard library, numpy and scipy only; no other varstar
  layer, no file/FITS/process I/O library.
- ``application``: use cases. May import ``domain`` (and its own ports), never ``adapters``
  or ``cli``, and no I/O library either.
- ``adapters``: may import ``domain`` and ``application``, never ``cli``.
- ``cli``: composition root, may import everything.

Modules still at the top of the package (``varstar.qa``, ``varstar.pipeline``…) are the
legacy flat layout being migrated: the layer rules do not apply to them yet, but nothing in
a layer may import them except ``cli`` and ``adapters`` (temporary).

Everywhere: no import of a private name (``_xxx``) from another module.

``TOLERATED`` lists the known violations being migrated away. The list must only shrink:
the test also fails when a tolerated violation has disappeared, so that it gets removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[2] / "src" / "varstar"
LAYERS = ("domain", "application", "adapters", "cli")

IO_LIBS = ("astropy.io", "subprocess", "concurrent.futures", "logging.handlers", "shutil", "typer", "yaml",
           "astroquery", "matplotlib")
FORBIDDEN = {
    "domain": ("varstar.application", "varstar.adapters", "varstar.cli", *IO_LIBS),
    "application": ("varstar.adapters", "varstar.cli", *IO_LIBS),
    "adapters": ("varstar.cli",),
    "cli": (),
}
# Which legacy (flat) modules a layer may still import during the migration.
MAY_IMPORT_LEGACY = {"domain": False, "application": False, "adapters": True, "cli": True}

# (importing module, imported name) — to be emptied by the end of the migration (step 8).
TOLERATED: set[tuple[str, str]] = {
    ("varstar.qa", "varstar.inventory._nearest_night"),
}


def module_name(path: Path) -> str:
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def layer_of(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 and parts[1] in LAYERS else None


def is_legacy(module: str) -> bool:
    parts = module.split(".")
    return parts[0] == "varstar" and len(parts) > 1 and parts[1] not in LAYERS


def resolve(module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    base = base[:len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def imports(path: Path) -> list[tuple[str, str | None]]:
    """(imported module, imported name or None) for every import statement of a file."""
    module = module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(alias.name, None) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            target = resolve(module, path.name == "__init__.py", node)
            out += [(target, alias.name) for alias in node.names]
    return out


def matches(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(prefix + ".")


def violations() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(PKG.rglob("*.py")):
        module = module_name(path)
        layer = layer_of(module)
        for target, name in imports(path):
            full = f"{target}.{name}" if name else target
            # A sub-module imported with `from package import module` counts as that module.
            if name and (PKG.parent / Path(*full.split("."))).with_suffix(".py").exists():
                target, name = full, None
            if name and name.startswith("_") and not name.startswith("__") and target.startswith("varstar") \
                    and target != module:
                found.add((module, f"{target}.{name}"))
            if layer is None:
                continue
            for prefix in FORBIDDEN[layer]:
                if matches(target, prefix):
                    found.add((module, target))
            if is_legacy(target) and not MAY_IMPORT_LEGACY[layer]:
                found.add((module, target))
    return found


def test_layer_rules():
    new = violations() - TOLERATED
    assert not new, "Règles d'architecture enfreintes :\n" + "\n".join(
        f"  {m} importe {t}" for m, t in sorted(new))


def test_tolerated_list_only_shrinks():
    gone = TOLERATED - violations()
    assert not gone, "Exceptions résolues, à retirer de TOLERATED :\n" + "\n".join(
        f"  {m} → {t}" for m, t in sorted(gone))


def test_checker_detects_violations(tmp_path, monkeypatch):
    """The checker itself: a fake domain module importing astropy.io and a legacy module."""
    pkg = tmp_path / "varstar"
    (pkg / "domain").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "domain" / "__init__.py").write_text("")
    (pkg / "qa.py").write_text("def _helper():\n    pass\n")
    (pkg / "domain" / "bad.py").write_text(
        "from astropy.io import fits\nfrom .. import qa\nfrom ..qa import _helper\n")
    monkeypatch.setattr(__import__(__name__), "PKG", pkg)
    assert violations() == {("varstar.domain.bad", "astropy.io"), ("varstar.domain.bad", "varstar.qa"),
                            ("varstar.domain.bad", "varstar.qa._helper")}
