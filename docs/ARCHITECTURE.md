# Architecture de varstar-pipeline

Architecture hexagonale pragmatique (ports et adaptateurs), sans framework d'injection.

Le cœur métier est pur et se teste sans disque. Chaque dépendance externe passe par un port
(`typing.Protocol`) : FITS, Siril, dossier de travail, catalogues… La ligne de commande est la
seule racine de composition.

Les règles ci-dessous sont **vérifiées automatiquement** par `tests/architecture/test_layers.py`.

## Les quatre couches

```
src/varstar/
  domain/        règles métier pures
  application/   cas d'usage + ports (Protocol)
  adapters/      implémentations des ports (fichiers, FITS, Siril, PixInsight, Markdown, YAML)
  cli/           commandes Typer + racine de composition (bootstrap.py)
  __init__.py    version ; defaults.yaml : configuration par défaut
```

**Règle de dépendance** : `domain` ← `application` ← `adapters` / `cli`.

| Couche | Peut importer | Interdit |
|---|---|---|
| `domain` | bibliothèque standard, numpy, scipy | toute autre couche, `astropy.io`, `subprocess`, `concurrent.futures`, `logging.handlers`, `shutil`, `typer`, `yaml`, `astroquery`, `matplotlib` |
| `application` | `domain`, ses propres ports | `adapters`, `cli`, les mêmes bibliothèques d'E/S |
| `adapters` | `domain`, `application` | `cli` |
| `cli` | tout | dans `cli/`, seul `bootstrap.py` importe les adaptateurs (sauf `adapters.reporting`, qui n'est que de la mise en forme) |

Autres règles :
- Aucun module ne vit hors des couches.
- Aucun import d'un nom privé (`_xxx`) d'un autre module.
- La liste `TOLERATED` du test est vide. Y ajouter une exception demande une issue GitHub qui la justifie.

## Où placer quoi

| Je dois écrire… | Couche | Exemple existant |
|---|---|---|
| une règle de décision (niveau A/B/C, appariement, contrôle qualité, rejet, masque) | `domain` | `domain/inventory.py`, `domain/qa/raw.py`, `domain/calibration.py` |
| un calcul sur des tableaux déjà lus | `domain` | `domain/qa/pixels.py`, `domain/masks.py` |
| un objet métier, un plan, un réglage typé | `domain` | `domain/model.py`, `domain/plan.py`, `domain/settings.py` |
| l'enchaînement d'une étape (lire → décider → publier, cache, provenance) | `application` | `application/masters.py`, `application/calibrate.py` |
| un besoin envers le monde extérieur | un **port** dans `application/ports.py` | `MasterBuilder`, `PixelStats`, `ArtifactStore` |
| la lecture ou l'écriture d'un fichier, l'appel d'un outil externe | `adapters` | `adapters/siril_masters.py`, `adapters/fits_io.py` |
| un rapport Markdown, JSON ou HTML en français | `adapters/reporting` | `adapters/reporting/qa_markdown.py` |
| une commande, une option, un message de la ligne de commande | `cli/app.py` | `varstar check` |
| le branchement d'un nouvel adaptateur | `cli/bootstrap.py` | `pipeline_services()` |

Repères pratiques :
- Le domaine reçoit des données déjà lues : `Frame`, `FrameSet`, statistiques par image, tableaux numpy, `FileStat` (empreinte d'un fichier). Il n'ouvre jamais de fichier.
- Les messages des anomalies et des contrôles (en français) font partie du domaine : ce sont des décisions métier. Leur mise en page (Markdown) appartient aux adaptateurs.
- Les cas d'usage n'affichent rien. Ils signalent leur progression par un `Progress` (`Reporter.progress`) et journalisent avec `logging.getLogger("varstar")`. Le fichier de journal, un par étape, est géré par `adapters/console.py`.
- Erreurs à montrer à l'utilisateur : `PipelineError` (domaine), `SirilError`, `SettingsError`. La ligne de commande les affiche en « ERREUR : … », sans trace Python.

## Ports

| Port | Rôle | Adaptateur | Faux pour les tests |
|---|---|---|---|
| `FrameSource` | fichiers FITS d'une source, en-têtes, empreinte d'un fichier | `FilesystemFrameSource` | `InMemoryFrameSource` |
| `PixelReader` | facteur ADU d'une image brute | `FitsPixelReader` | `FakePixelReader` |
| `PixelStats` | statistiques rapides, flats binnés, lecture d'images (avec cache) | `PixelStatsCache` | `FakePixelStats` |
| `ArtifactStore` | dossier de travail : JSON, texte, YAML, images, masques | `Workspace` | `InMemoryArtifactStore` |
| `ProvenanceRecorder` | empreintes (cache), manifestes, `provenance.json` | `ProvenanceStore` | `FakeProvenance` |
| `MasterRepository` | bibliothèque de masters partagée | `DiskMasterRepository` | `InMemoryMasterRepository` |
| `MasterBuilder` | construction des masters | `SirilMasterBuilder` | `FakeMasterBuilder` |
| `Calibrator` | calibration d'un groupe de lights | `SirilCalibrator` | `FakeCalibrator` |
| `ReferenceMasters` | masters PixInsight (validation croisée) | `PixInsightMasters` | `FakeReferenceMasters` |
| `InventoryReport`, `QAReport`, `MastersReport` | `inventory.json/md`, `qa.md`, `masters.md` | `adapters/reporting/*` | `Recording*` |
| `Reporter` | progression affichée à l'utilisateur | `ConsoleReporter` | `RecordingReporter` |
| `StarCatalog` | étoiles de référence (Gaia DR3 par TAP avec cache disque, ou CSV local) | `GaiaTapCatalog`, `CsvStarCatalog` | `FakeStarCatalog` |
| `PlateSolver` | WCS des images calibrées (séries en parallèle) | `InternalPlateSolver` | `FakePlateSolver` |
| `AstrometryReport` | `astrometry.md` | `AstrometryReportAdapter` | `RecordingAstrometryReport` |

Les faux sont dans `tests/fakes.py`. Chacun passe les mêmes **tests de contrat** que l'adaptateur
réel (`tests/contracts/`), avec le vrai Siril sous le marqueur `siril`. Les tests de cas d'usage
peuvent donc s'appuyer sur eux.

## Configuration

- `adapters/config_loader.py` lit les défauts (`defaults.yaml`), le fichier de l'utilisateur et d'éventuelles surcharges.
- Il produit des `Settings` figés (`domain/settings.py`), avec une dataclass par section. Une clé inconnue, absente ou de mauvais type donne une `SettingsError` en français qui nomme la clé.
- Une fonction reçoit la seule section utile quand une suffit (`MasksSettings`, `MastersSettings`, `MatchingSettings`), sinon `Settings`.
- `Settings.raw`, la configuration fusionnée et jamais modifiée, est écrite telle quelle dans `config.used.yaml` et `provenance.json`. C'est aussi elle que hachent les empreintes de cache.
- Une nouvelle section (par exemple `photometry` au jalon M4) devient une dataclass dans `TYPED_SECTIONS` le jour où du code l'utilise.

## Ajouter un jalon (M3 et suivants) : la marche à suivre

Exemple : l'astrométrie (faite au jalon M3 en suivant ces étapes : `domain/astrometry.py`, `domain/stars.py`, `domain/qa/astrometry.py`, `application/astrometry.py`, `adapters/plate_solver.py`, `adapters/star_catalog.py`).
1. **Domaine** : les décisions et les calculs purs, comme la validation d'une solution WCS ou l'échelle attendue à partir de `FOCALLEN` et `XPIXSZ`, avec leurs tests dans `tests/domain/` (hypothesis si la règle s'y prête, voir plus bas).
2. **Port** : `PlateSolver` dans `application/ports.py`, et son faux dans `tests/fakes.py`.
3. **Cas d'usage** : `application/astrometry.py` (cache, provenance, publication via `ArtifactStore`), testé avec les faux dans `tests/application/`.
4. **Adaptateur** : `adapters/astap.py` (ou astrometry.net), avec un test de contrat faux/réel dans `tests/contracts/`, sous un marqueur s'il faut l'outil installé.
5. **Branchement** : l'étape dans `application/pipeline.py` (`STEPS`, `IMPLEMENTED`, `PipelineServices`) et l'adaptateur dans `cli/bootstrap.py`.
6. **Bout en bout** : étendre `tests/e2e/minisky.py` (WCS synthétique, faux catalogue), puis mettre à jour les références avec `pytest --update-golden`, en expliquant le changement dans le commit.

## Tests

| Dossier | Ce qu'on teste | Dépendances |
|---|---|---|
| `tests/domain/` | règles métier, cas limites, propriétés | `builders.py`, sans disque |
| `tests/application/` | orchestration : appels des ports, cache, erreurs | `builders.py` + `fakes.py` |
| `tests/adapters/` | conversion fidèle (FITS, Siril, Markdown, YAML) | `tmp_path`, `fitsgen.py` |
| `tests/contracts/` | le faux se comporte comme le réel | les deux implémentations |
| `tests/e2e/` | la ligne de commande complète sur la mini-session `minisky` (références dans `golden/`) ; données réelles sous `real_data` | Siril sous `siril` |
| `tests/architecture/` | les règles de ce document | analyse `ast` |

- **Couverture** : `tools/check_coverage.sh` est bloquant, avec domaine ≥ 90 % et application ≥ 85 %.
- **Hypothesis**, seulement pour une fonction pure du domaine qui remplit trois conditions :
  - un grand espace d'entrées ;
  - une propriété qui ne réimplémente pas l'algorithme (invariance, partition, aller-retour, bornes) ;
  - une exécution rapide.

  Tout contre-exemple est épinglé avec `@example`.
