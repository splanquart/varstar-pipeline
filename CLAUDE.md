# varstar-pipeline

Pipeline automatique : **images brutes N.I.N.A. → calibration (Siril) → astrométrie → photométrie différentielle → détection d'étoiles variables → rapport**, lancé par une seule commande.

## À lire avant de coder
1. `docs/SPEC.md` — spécification fonctionnelle et technique (référence).
2. `docs/DECISIONS.md` — choix déjà tranchés et leur justification, questions ouvertes.
3. `docs/DATA_NGC1499_2021.md` — description du jeu de données de référence (tests réels).
4. `config.example.yaml` — paramètres par défaut.
5. `docs/QA_CHECKS.md` — contrôles qualité précoces à intégrer à chaque étape (détecter les erreurs d'acquisition avant le calcul des flux).
6. `docs/ARCHITECTURE.md` — couches, ports, où placer quoi, marche à suivre pour un nouveau jalon.

Construire par jalons (SPEC §10), dans l'ordre. À la fin de chaque jalon : tests verts + exécution sur la session `sept06` du jeu de référence, et un court compte rendu.

## Environnement
- macOS, Python ≥ 3.11, environnement virtuel dédié.
- Siril installé (version récente). Avant d'écrire le moindre script Siril : `siril-cli --version` et vérifier la syntaxe des commandes dans la documentation **de la version installée** (la syntaxe de `calibrate`/`stack` a évolué entre versions). Chemin macOS habituel : `/Applications/Siril.app/Contents/MacOS/siril-cli`.
- Bibliothèques : numpy, astropy, photutils, astroquery, pandas, matplotlib, pyyaml, typer (CLI), pytest.

## Règles non négociables (photométrie)
- **Les fichiers bruts sont en lecture seule.** Ne jamais les déplacer, renommer, modifier ou supprimer. Tout est écrit dans le dossier de travail.
- **Pas de stacking des lights, pas d'alignement/rééchantillonnage, pas de drizzle, pas de normalisation locale, pas de débayérisation.** La photométrie se fait image par image, sur les images calibrées non alignées, en plaçant les ouvertures grâce à la WCS de chaque image.
- **Pas de correction cosmétique qui modifie les pixels** : on construit un masque de pixels défectueux et on *marque* les mesures touchées.
- **Calibration stratégie B (capteur CMOS)** : dark de même pose/gain/binning/température ; flats − dark de flat ; pas de bias ; pas de mise à l'échelle des darks sauf repli explicite (niveau B, journalisé).
- **Jamais de dégradation silencieuse** : chaque session reçoit un niveau de calibration (A/B/C) visible dans le rapport.
- Calculs en **float32** minimum ; masters en FITS 32 bits.
- Unités : les fichiers N.I.N.A. de l'ASI294MM sont en uint16 avec des valeurs multiples de 4 (14 bits × 4). Gain effectif ≈ `EGAIN / 4` e⁻/ADU — le calculer à partir des données (détection du facteur d'échelle), ne pas le coder en dur.
- Reproductibilité : chaque exécution écrit la config utilisée, les versions logicielles et la provenance de chaque master/image dans `provenance.json`.

## Conventions
- Code, noms de variables, docstrings : anglais. Messages à l'utilisateur, logs de haut niveau et rapports : **français**.
- Découverte des fichiers **par les en-têtes FITS**, jamais par l'arborescence (les noms de dossiers peuvent être faux).
- Les noms de fichiers N.I.N.A. contiennent des espaces (`NGC 1499`) : Siril doit travailler sur des liens symboliques aux noms sûrs.
- Tests unitaires sur des FITS synthétiques générés par les tests (aucune dépendance au jeu réel de 15 Go). Tests d'intégration réels marqués `@pytest.mark.real_data` et désactivés par défaut.
- Étapes idempotentes avec cache : relancer ne recalcule que ce qui a changé.
- Problèmes repérés en chemin : issues GitHub de `splanquart/varstar-pipeline` (`gh issue create`, en français : symptôme, cause, impact, piste). Les citer par leur URL dans le code ou les tests.

## Architecture (hexagonale, vérifiée par `tests/architecture/test_layers.py`)
- `domain/` (règles pures : stdlib, numpy, scipy ; aucune E/S) ← `application/` (cas d'usage + ports `Protocol` dans `ports.py`) ← `adapters/` (FITS, Siril, fichiers, Markdown, YAML) / `cli/` (commandes ; `cli/bootstrap.py` est la **seule** racine de composition).
- Nouveau besoin externe (solveur astrométrique, Gaia, VSX…) : un port dans `application/ports.py`, un faux dans `tests/fakes.py`, un adaptateur, un test de contrat faux/réel dans `tests/contracts/`, le branchement dans `cli/bootstrap.py`. Détail : `docs/ARCHITECTURE.md`.
- Pas de `print` dans les couches (progression : `Reporter`/`Progress`), configuration typée (`domain/settings.py`, une dataclass figée par section ; `Settings.raw` pour `config.used.yaml` et les empreintes).
- Tests par couche (`tests/domain`, `application`, `adapters`, `contracts`, `e2e`, `architecture`) ; couverture bloquante (domaine ≥ 90 %, application ≥ 85 %).

## Commandes
Disponibles : `inventory`, `check`, `darks-todo`, `run` (étapes `inventory`, `check`, `masters`, `calibrate`, `astrometry`). Environnement : `.venv/` (installer avec `pip install -e .`). Tests Siril de bout en bout dans `tests/e2e/` (marqueur `siril`, sautés si `siril-cli` est absent).
```
varstar inventory <dossier>            # inventaire + niveaux de calibration, sans rien calculer
varstar check <dossier> [--headers-only] # contrôles qualité précoces (qa/qa.md), dès le lendemain d'une séance
varstar run <dossier> [-c config.yaml] # pipeline complet
varstar run <dossier> --from photometry  # reprise à une étape
varstar darks-todo <dossier>           # liste des darks manquants à refaire
pytest                                  # tests unitaires + e2e « minisky » (mini-images synthétiques)
pytest -m siril                         # seulement les tests qui appellent le vrai siril-cli (sautés s'il est absent)
pytest -m real_data                     # tests sur données réelles
pytest -m network                       # tests qui interrogent les services Gaia (internet)
pytest --update-golden                  # réécrit tests/e2e/golden/ et tests/snapshots/ (changement voulu : l'expliquer dans le commit)
tools/check_coverage.sh                 # couverture par couche, seuils bloquants (ENFORCE=0 : rapport seulement)
python tools/compare_outputs.py capture <workdir>             # instantané de recette (sept06) → <workdir>/_baseline
python tools/compare_outputs.py compare <workdir>/_baseline <workdir>
```
Générateurs de FITS de test : `tests/fitsgen.py` (`write_frame`, `series`) ; jeu e2e : `tests/e2e/minisky.py`.
