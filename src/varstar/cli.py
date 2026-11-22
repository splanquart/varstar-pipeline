"""Command-line interface (`varstar ...`). User-facing messages are in French."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_config, resolve_workdir
from .inventory import build_inventory, darks_todo
from .inventory_io import todo_markdown
from .steps import run_inventory

app = typer.Typer(add_completion=False, help="Pipeline de détection d'étoiles variables (N.I.N.A. → rapport).")

SourceArg = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, help="Dossier des images brutes")
ConfigOpt = typer.Option(None, "--config", "-c", exists=True, dir_okay=False, help="Fichier de configuration YAML")
WorkdirOpt = typer.Option(None, "--workdir", "-w", help="Dossier de travail (sorties)")


@app.command()
def inventory(source: Path = SourceArg, config: Optional[Path] = ConfigOpt,
              workdir: Optional[Path] = WorkdirOpt,
              force: bool = typer.Option(False, "--force", help="Recalculer même si rien n'a changé")) -> None:
    """Inventaire des fichiers et niveaux de calibration, sans aucun calcul sur les images."""
    cfg = load_config(config)
    wd = resolve_workdir(source, cfg, workdir)
    res = run_inventory(source, cfg, wd, force=force)
    data = res.data
    if res.skipped:
        typer.echo("Inventaire inchangé depuis la dernière exécution (utiliser --force pour recalculer).")
    counts = data["counts"]
    typer.echo(f"Fichiers retenus : " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    typer.echo("Niveaux de calibration (session / filtre) :")
    for item in data["levels"]:
        typer.echo(f"  {item['label']:<24} {item['filter']:<4} {item['level']}")
    n_warn = sum(1 for a in data["anomalies"] if a["severity"] in ("warning", "error"))
    typer.echo(f"Anomalies : {n_warn} avertissement(s)/erreur(s), "
               f"{len(data['anomalies']) - n_warn} info(s)")
    typer.echo(f"Darks à refaire : {len(data['darks_todo'])}")
    typer.echo(f"Rapport : {wd / 'inventory.md'}")


@app.command()
def run(source: Path = SourceArg, config: Optional[Path] = ConfigOpt, workdir: Optional[Path] = WorkdirOpt,
        from_step: str = typer.Option("inventory", "--from", help="Étape de reprise"),
        sessions: Optional[str] = typer.Option(None, "--sessions", help="ex. sept06,2021-09-12"),
        filters: Optional[str] = typer.Option(None, "--filters", help="ex. H,S"),
        force: bool = typer.Option(False, "--force", help="Ignorer les caches de l'espace de travail")) -> None:
    """Pipeline complet (étapes disponibles : inventory, check, masters, calibrate)."""
    from .pipeline import PipelineError, run_pipeline
    from .siril import SirilError

    cfg = load_config(config)
    wd = resolve_workdir(source, cfg, workdir)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None  # noqa: E731
    typer.echo(f"Dossier de travail : {wd}")
    try:
        summary = run_pipeline(source, cfg, wd, from_step, split(sessions), split(filters), force=force)
    except (PipelineError, SirilError) as exc:
        typer.echo(f"ERREUR : {exc}", err=True)
        raise typer.Exit(1)
    plan = summary["plan"]
    typer.echo(f"Masters : {len(plan.get('masters', {}))} — rapport {wd / 'masters' / 'masters.md'}")
    cal = summary.get("calibrate")
    if cal:
        n = sum(len(r["outputs"]) for r in cal["results"].values())
        skipped = sum(1 for r in cal["results"].values() if r.get("skipped"))
        typer.echo(f"Images calibrées : {n} ({skipped} groupe(s) déjà à jour) → {wd / 'calibrated'}")
        for key, err in cal["errors"].items():
            typer.echo(f"  ÉCHEC {key} : {err}", err=True)
    for d in summary.get("qa", {}).get("header", []) + summary.get("qa", {}).get("raw", []) + (
            cal.get("qa", []) if cal else []):
        if d["severity"] != "info":
            typer.echo(f"  Contrôle {d['check']} : {d['message']}")
    if summary.get("qa") or (cal and cal.get("qa")):
        typer.echo(f"Contrôles qualité : {wd / 'qa' / 'qa.md'}")
    for w in plan.get("warnings", []):
        typer.echo(f"  Avertissement : {w}")
    if summary["not_implemented"]:
        typer.echo("Étapes suivantes pas encore disponibles : " + ", ".join(summary["not_implemented"]))
    if cal and cal["errors"]:
        raise typer.Exit(2)


@app.command()
def check(source: Path = SourceArg, config: Optional[Path] = ConfigOpt, workdir: Optional[Path] = WorkdirOpt,
          headers_only: bool = typer.Option(False, "--headers-only", help="En-têtes seulement (aucun pixel lu)")) -> None:
    """Contrôles qualité précoces (en-têtes + pixels bruts), à lancer dès le lendemain d'une séance."""
    from .qa import SEVERITY_FR, all_items, run_checks, summary_line

    cfg = load_config(config)
    wd = resolve_workdir(source, cfg, workdir)
    wd.mkdir(parents=True, exist_ok=True)
    from .steps import setup_logging
    setup_logging(wd, "check", cfg["runtime"].get("log_level", "INFO"))
    inv = build_inventory(source, cfg, exclude=[wd])
    report = run_checks(inv, cfg, wd, raw=not headers_only)
    items = all_items(report, ("header",) if headers_only else ("header", "raw"))
    typer.echo(f"Contrôles qualité : {summary_line(items)}")
    for i in items:
        if i.severity != "info":
            typer.echo(f"  [{i.check} {SEVERITY_FR[i.severity]}] {i.message}")
            if i.advice:
                typer.echo(f"      → {i.advice}")
    typer.echo(f"Rapport : {wd / 'qa' / 'qa.md'}")
    if any(i.severity == "error" for i in items):
        raise typer.Exit(3)


@app.command("darks-todo")
def darks_todo_cmd(source: Path = SourceArg, config: Optional[Path] = ConfigOpt) -> None:
    """Liste des darks (et darks de flat) manquants à refaire."""
    cfg = load_config(config)
    inv = build_inventory(source, cfg)
    typer.echo("\n".join(todo_markdown(darks_todo(inv))))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
