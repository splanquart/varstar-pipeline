# Spécification — varstar-pipeline

## 1. Objectif
Avec une seule commande, passer des images brutes d'une cible (plusieurs sessions, plusieurs filtres) à :
- des courbes de lumière différentielles de **toutes** les étoiles mesurables du champ ;
- une liste d'étoiles **candidates variables**, croisée avec le catalogue VSX (variables connues / nouvelles candidates) ;
- un rapport lisible (en français) qui expose aussi la qualité de calibration de chaque session.

L'utilisateur continue à traiter ses « belles images » dans PixInsight. Ce pipeline est une chaîne **scientifique** séparée.

## 2. Périmètre
**v1** : caméra monochrome (ASI294MM), acquisition N.I.N.A., FITS bruts, filtres quelconques (y compris bandes étroites Hα/OIII/SII), photométrie d'ouverture forcée, détection de variabilité, rapport.

**Hors v1** : caméras couleur, photométrie PSF, magnitudes standard (transformation vers V/B…), export AAVSO (les filtres étroits ne sont pas acceptés par l'AAVSO ; à prévoir quand des filtres photométriques seront utilisés), interface graphique.

## 3. Entrées
- Un dossier racine (ex. `NGC1499/YEAR_2021`) exploré récursivement.
- Seuls les `*.fits` / `*.fit` / `*.fts` sont considérés. Ignorés par défaut (configurable) : dossiers `removed/`, `output/`, `*.pxiproject/`, fichiers `.xisf`, `.tif`, `.jpg`.
- Classification **par en-têtes** :
  - `IMAGETYP` (LIGHT / DARK / FLAT / BIAS ; tolérer les variantes « Light Frame », « Dark Frame »…),
  - `EXPTIME` (ou `EXPOSURE`), `GAIN`, `OFFSET` (souvent absent → considéré constant, avertissement unique), `XBINNING`/`YBINNING`, `NAXIS1/2`, `CCD-TEMP`, `SET-TEMP`, `FILTER`, `DATE-OBS` (UTC), `OBJECT`, `RA`/`DEC`, `FOCALLEN`, `XPIXSZ`, `EGAIN`, `SITELAT`/`SITELONG`/`SITEELEV`, `AIRMASS`, `INSTRUME`, `TELESCOP`.
- Un fichier dont l'en-tête est incohérent avec son dossier (ex. dark de 180 s rangé dans `300s/`) est classé selon son en-tête, avec un avertissement.

## 4. Pipeline

### 4.1 Inventaire (`varstar inventory`)
- Lire uniquement les en-têtes (rapide, sans charger les pixels).
- **Session** = nuit d'observation : date de `DATE-OBS` décalée de −12 h (une nuit du 6 au 7 = session du 6). Les sous-dossiers `SESSION_xxx` éventuels servent seulement de libellés.
- **Clé de configuration capteur** : (instrument, gain, offset, binning, dimensions). Tout appariement exige une clé identique.
- Appariement :
  - light ↔ dark : même clé capteur, même `EXPTIME` (tolérance 0,01 s), |ΔT| ≤ `temp_tolerance` (défaut 1,5 °C sur la moyenne du groupe) ;
  - flat ↔ light : même filtre ; priorité aux flats de la même session ; sinon flats de la session la plus proche dans le temps (≤ `max_flat_age_days`) **si** `same_optical_setup` est confirmé dans la config (voir DECISIONS) ;
  - flat ↔ dark de flat : même clé capteur, même `EXPTIME` (tolérance 0,05 s) ; sinon le plus proche (niveau B).
- Attribution d'un **niveau de calibration** par (session, filtre) :

| Niveau | Darks des lights | Flats | Dark de flat |
|---|---|---|---|
| A | même pose, gain, binning, ΔT ok | même session | même pose |
| B | pose différente → dark mis à l'échelle à partir de darks courts | session voisine, même montage déclaré | pose la plus proche |
| C | aucun → seulement masque de pixels chauds | aucun → courbes valables **intra-session uniquement** | — |

- Sorties : `inventory.md` (tableau lisible), `inventory.json` (machine), liste des anomalies.
- `varstar darks-todo` : liste des combinaisons (pose, gain, binning, température) pour lesquelles il manque des darks, pour pouvoir les refaire après coup.

### 4.2 Masters (via Siril)
- Bibliothèque de masters en cache, indexée par leurs paramètres (réutilisable entre cibles et années) : `library/darks/`, `library/flats/`.
- Combinaison par **moyenne avec rejet**, algorithme choisi selon le nombre d'images N :
  - N < 8 : rejet par percentile (ou médiane) ;
  - 8 ≤ N < 15 : sigma clipping 3/3 ;
  - 15 ≤ N ≤ 25 : winsorized sigma clipping 3/3 ;
  - N > 25 : linear fit clipping (ou GESD).
- Masters darks : aucune normalisation.
- Masters flats : chaque flat − master dark de flat, puis normalisation **multiplicative**, puis combinaison avec rejet. Vérifier que le niveau médian des flats est entre 15 % et 85 % de la dynamique, sinon avertissement.
- Masters en FITS float32, avec en-têtes de provenance (liste des fichiers sources, algorithme, N, date).
- Siril est piloté par des scripts `.ssf` générés, exécutés par `siril-cli -s script.ssf -d <workdir>`. Travailler sur des liens symboliques aux noms sans espaces. Capturer stdout/stderr dans les logs ; échec Siril = erreur claire avec la commande fautive.
- **Validation croisée (facultative)** : si des masters PixInsight existent (`output/master/*.xisf`), comparer statistiquement nos masters aux leurs (médiane, écart-type de la différence) et l'indiquer dans le rapport.

### 4.3 Calibration des lights (via Siril)
- light_cal = (light − master_dark) / master_flat_normalisé, en float32.
- Pas d'optimisation de dark (`-opt`) en niveau A ; activée uniquement en niveau B pour les darks.
- **Pas de correction cosmétique** dans Siril : on construit un **masque de pixels défectueux** (BPM) depuis le master dark (pixels chauds > k·σ, pixels froids) et depuis le master flat (pixels morts), stocké à part.
- Masque des **coins d'amp glow** (zones où le master dark dépasse un seuil de gradient) : les étoiles dans ces zones sont marquées.

### 4.4 Astrométrie
- Une WCS par image calibrée, **sans rééchantillonner l'image** (la WCS est seulement ajoutée à l'en-tête ou stockée à côté).
- Moteur : solveur de Siril (`platesolve` / `seqplatesolve` selon la version) en premier choix ; **ASTAP** en ligne de commande en repli (configurable). Indices fournis : `RA`, `DEC`, `FOCALLEN`, `XPIXSZ` × binning.
- Échelle attendue pour le matériel de référence : ≈ 3,49″/px, champ ≈ 4,0° × 2,7°.
- Image non résolue → exclue, et listée dans le rapport.
- Déduire de l'angle de rotation de la WCS un **segment** (orientation 0° / 180° = avant / après retournement au méridien). Un changement d'orientation coupe la session en segments indépendants pour la photométrie.

### 4.5 Qualité des images
Pour chaque image : FWHM médiane, ellipticité, fond de ciel, nombre d'étoiles détectées, point zéro instrumental (étape 4.7). Rejet automatique (configurable) si FWHM > `fwhm_max_factor` × médiane de la session, si le nombre d'étoiles chute fortement (nuages) ou si le point zéro chute de plus de `zp_drop_max` mag. Les images rejetées sont listées avec la raison.

### 4.6 Catalogue de référence des étoiles
- Requête **Gaia DR3** (astroquery/VizieR) sur le champ commun des images, jusqu'à une magnitude limite configurable (défaut : G < 15, à ajuster selon la profondeur réelle).
- Chaque étoile garde son identifiant Gaia, RA/Dec (époque corrigée du mouvement propre à la date moyenne des observations), G, BP-RP.
- Signalement des **voisines** : une étoile est marquée « contaminée » si une autre étoile Gaia plus brillante que (G + Δmag) se trouve dans l'anneau de ciel ou à moins de 2 rayons d'ouverture.

### 4.7 Photométrie d'ouverture forcée
- Positions : RA/Dec Gaia → pixels via la WCS de chaque image, puis recentrage fin optionnel (centroïde limité à ±1 px) pour absorber les petites erreurs de WCS.
- Rayon d'ouverture = `k_aperture` × FWHM de l'image (défaut 1,5) ; mesurer aussi à 1,0 et 2,0 × FWHM pour choisir après coup le meilleur rayon par plage de magnitude.
- Ciel : anneau de `annulus_in` à `annulus_out` × FWHM (défaut 4–7), médiane sigma-clippée. Important : le fond Hα de NGC 1499 est structuré, l'anneau doit rester local.
- Erreur : équation CCD avec gain effectif (EGAIN / facteur d'échelle 14→16 bits), bruit du ciel, bruit de lecture (paramètre de config).
- Marqueurs par mesure : `saturated` (pic brut > `saturation_frac` × valeur max), `edge`, `bad_pixel` (BPM dans l'ouverture), `amp_glow`, `contaminated`, `low_snr`.
- Temps : milieu de pose (`DATE-OBS` + `EXPTIME`/2), converti en **BJD_TDB** avec la position du site (`SITELAT`/`SITELONG`) et les coordonnées de l'étoile. Conserver aussi le JD UTC.

### 4.8 Photométrie différentielle (par filtre)
- Traitement **séparé par filtre**.
- Point zéro par image à partir d'un **ensemble** d'étoiles de comparaison :
  1. candidates initiales : étoiles non marquées, SNR > seuil, non variables connues VSX ;
  2. point zéro de l'image = moyenne pondérée robuste des écarts (mag instrumentale − mag moyenne de l'étoile) ;
  3. itérations : on retire de l'ensemble les étoiles dont la dispersion est anormale, on recalcule (3 à 5 itérations, convergence).
- Correction optionnelle d'un terme de position dans le champ (gradient linéaire ou quadratique du point zéro, pour absorber les résidus de flat) — activée par défaut pour les sessions de niveau C.
- Un **point zéro par segment** absorbe les décalages entre sessions et entre segments.
- Résultat par mesure : magnitude différentielle, erreur, marqueurs, session, segment, niveau de calibration.

### 4.9 Détection de variabilité
Pour chaque étoile et chaque filtre (minimum `min_points` mesures valides) :
- magnitude médiane, dispersion robuste (1,4826 × MAD), erreur médiane attendue ;
- χ² réduit par rapport à une constante ;
- indice **Stetson J** (paires de mesures proches dans le temps ; l'utilisateur alterne les filtres H/S ou O/S toutes les ~5 min, donc paires inter-filtres possibles) ;
- rapport de von Neumann (1/η) — sensible aux variations lentes ;
- amplitude intra-nuit et amplitude entre nuits, séparément.

Critère de candidature (configurable) :
- dispersion robuste au-dessus de l'enveloppe « dispersion vs magnitude » : on ajuste la médiane glissante et sa dispersion, et une étoile est candidate si elle dépasse de `n_sigma_envelope` (défaut 5) ;
- **et** variabilité confirmée par au moins un critère indépendant : plusieurs filtres, plusieurs nuits, ou Stetson J élevé.
- Les variations visibles **uniquement entre sessions de niveau C** sont rétrogradées (« à confirmer »).

Pour chaque candidate : périodogramme de Lomb–Scargle (avertir sur les alias liés à l'échantillonnage nocturne et à la faible couverture), meilleure période si significative, courbe repliée.

Croisement **VSX** (VizieR `B/vsx/vsx`, rayon ~2 px ≈ 7″) : variable connue (type, période, amplitude VSX) ou **nouvelle candidate**. Contrôle de cohérence : les variables VSX du champ assez brillantes doivent ressortir ; lister celles qui ne ressortent pas (amplitude trop faible ? étoile marquée ?).

### 4.10 Sorties
```
<workdir>/
  inventory.md / inventory.json
  library/            masters (cache partagé, peut être hors workdir)
  calibrated/<session>/<filtre>/*.fits   (float32 + WCS)
  masks/              BPM, amp glow
  photometry/raw_measurements.parquet
  photometry/lightcurves.parquet         (toutes les étoiles, différentiel)
  results/candidates.csv
  results/known_variables.csv
  results/plots/      RMS-mag par filtre, qualité des images, courbes des candidates (brutes + repliées)
  report.html         rapport auto-contenu en français
  provenance.json
  logs/
```
Le rapport contient : résumé, inventaire et niveau par session, images rejetées et pourquoi, qualité (FWHM, fond, point zéro au fil du temps), diagrammes RMS-magnitude, tableau des candidates (Gaia ID, RA/Dec, mag médiane, amplitude, indices, période, statut VSX), vignettes et courbes de lumière.

## 5. Configuration
Voir `config.example.yaml`. Toute valeur a un défaut raisonnable ; la config effective est copiée dans le dossier de travail.

## 6. CLI
- `varstar inventory <dossier>` — inventaire seul.
- `varstar run <dossier> [-c config] [--workdir chemin] [--from étape] [--sessions s1,s2] [--filters H,S]`.
- `varstar darks-todo <dossier>`.
- `varstar report <workdir>` — régénère le rapport.
Étapes nommées : `inventory`, `masters`, `calibrate`, `astrometry`, `quality`, `catalog`, `photometry`, `differential`, `detect`, `report`.

## 7. Robustesse
- Idempotence : chaque étape écrit un manifeste (entrées + hash de config) ; si rien n'a changé, elle est sautée.
- Une image ou une session en erreur n'arrête pas le pipeline : elle est exclue et signalée.
- Journalisation : un log par étape + résumé en fin d'exécution.

## 8. Performance
- Image 4144 × 2822 en float32 ≈ 47 Mo. Traiter image par image, en parallèle (`multiprocessing`, nombre de processus configurable), sans charger toute la série en mémoire.
- Le jeu de référence (~15 Go de bruts, 170 lights) doit tourner en une nuit sur un MacBook Pro ; viser bien moins.

## 9. Validation
- **Tests unitaires** sur FITS synthétiques : appariement darks/flats et niveaux A/B/C, conversion d'unités (facteur ×4), photométrie d'étoiles gaussiennes de flux connu, point zéro d'ensemble, indices de variabilité sur séries simulées (constante, sinusoïde, éclipse).
- **Test d'injection** sur données réelles : injecter une étoile artificielle variable (amplitude connue) dans des images calibrées et vérifier qu'elle est détectée ; en déduire l'amplitude minimale détectable selon la magnitude.
- **Contrôles réels** : plancher de dispersion des étoiles brillantes (quelques millimagnitudes attendus pour les meilleures) ; variables VSX connues du champ retrouvées.

## 9 bis. Contrôles qualité précoces
Voir `docs/QA_CHECKS.md` : contrôles sur les en-têtes, sur les bruts puis après calibration, chacun avec un conseil pour la prochaine séance ; section « Contrôles qualité » en tête des rapports ; commande `varstar check` et étape `check` du pipeline (entre `inventory` et `masters`), résultats dans `qa/qa.json` et `qa/qa.md`. Darks : même nuit uniquement (DECISIONS D16).

## 10. Jalons
1. **M1 — Inventaire** : lecture des en-têtes, sessions, appariements, niveaux A/B/C, `inventory.md`, `darks-todo`.
2. **M2 — Masters + calibration via Siril** : bibliothèque, scripts `.ssf`, BPM, masque d'amp glow, comparaison avec les masters PixInsight.
3. **M3 — Astrométrie** : WCS par image, segments, rejet des échecs.
4. **M4 — Catalogue + photométrie forcée** : Gaia, ouvertures, marqueurs, BJD.
5. **M5 — Différentiel + détection** : ensemble, points zéro, indices, candidates, VSX, Lomb–Scargle.
6. **M6 — Rapport** HTML + graphiques.
7. **M7 — Jeu complet** NGC 1499 2021, test d'injection, réglages par défaut finaux.
