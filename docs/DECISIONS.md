# Décisions et questions ouvertes

## Décisions prises

**D1 — Siril pour la calibration, Python pour le reste.**
L'utilisateur a installé Siril : sa calibration est équivalente à celle de PixInsight (même calcul, rejet comparable) et elle s'automatise entièrement en ligne de commande. Python orchestre tout et fait l'astrométrie (si nécessaire), la photométrie, la détection et le rapport. PixInsight reste l'outil des images esthétiques, en dehors de ce projet.

**D2 — Stratégie de calibration B (CMOS).**
Darks de même pose, sans bias, sans mise à l'échelle ; flats corrigés de darks de flat. Raison : l'IMX294 a un amp glow marqué qui ne se met pas correctement à l'échelle avec le temps de pose. La mise à l'échelle n'est qu'un repli (niveau B), toujours signalé.

**D3 — Pas d'alignement ni de stacking des lights.**
L'interpolation modifie le flux et le bruit. Photométrie sur les images calibrées non alignées ; les positions viennent de la WCS de chaque image.

**D4 — Pas de correction cosmétique : masque de pixels défectueux.**
Remplacer des pixels dans une étoile fausse la mesure. On marque les mesures touchées au lieu de les « réparer ».

**D5 — Photométrie forcée sur les positions Gaia DR3.**
Liste d'étoiles stable d'une image à l'autre, identifiants universels, détection des voisines contaminantes, positions précises même pour les étoiles faibles.

**D6 — Analyse filtre par filtre.**
Les bandes étroites ne sont pas des bandes standard ; aucune comparaison avec des magnitudes publiées. La détection est *différentielle*. Une variabilité vue dans plusieurs filtres renforce une candidate.

**D7 — Niveaux de calibration A/B/C par session et filtre.**
Les anciennes sessions sont incomplètes. On exploite tout, mais en rendant visible la qualité de chaque session ; les variations entre sessions de niveau C sont rétrogradées.

**D8 — Bruts en lecture seule, sorties dans un dossier de travail séparé.**

**D9 — Rapports en français, code en anglais.**

**D10 — Bibliothèque de masters partagée.**
Les masters de darks sont indexés par leurs paramètres et réutilisés entre cibles et années. Des darks refaits aujourd'hui avec les mêmes réglages pourront servir aux anciennes sessions (le niveau passe alors de B/C à A).

**D11 — Session sept02 : flats de sept06 (ancienne question Q1, confirmée le 21/11/2026).**
Aucune modification optique entre le 02/09 et le 07/09/2021. `config.yaml` déclare `same_optical_setup: [[2021-09-02, 2021-09-07]]` → sept02 utilise les flats de sept06, niveau **B**.

**D12 — Unités des images calibrées : ADU16 en float32.**
Siril écrit des flottants normalisés sur [0, 1] (= ADU16 / 65535, vérifié sur données synthétiques ; pas de retournement d'image, valeurs négatives conservées). Les masters gardent ce format (entrée directe de Siril) avec le mot-clé `VSNORM = 65535` ; les lights calibrés sont reconvertis en ADU16 (`BUNIT = 'ADU'`), reçoivent l'en-tête brut complet et la provenance (`VSRAW`, `VSLEVEL`, `VSDARK`, `VSFLAT`, `VSBPM`, `VSAMPG`, `EGAINEFF`).

**D13 — Siril isolé des préférences de l'utilisateur.**
`siril-cli -i <workdir>/siril/siril.ini` : les réglages imposés (32 bits, extension `.fits`) ne modifient pas la configuration Siril personnelle.
Sous macOS avec Siril ≥ 1.4.1, chaque script commence par `pyscript /dev/null`, juste après `requires`. Ce contournement de free-astro/siril#2021 force Siril à attendre la fin de sa vérification Python avant de quitter : sinon le Python laissé orphelin est tué par macOS et la fenêtre « Python a quitté de manière imprévue » s'affiche ([#1](https://github.com/splanquart/varstar-pipeline/issues/1)). Le réglage `siril.python_init_guard` vaut `auto`, `on` ou `off`. Si `pyscript` échoue, le script est relancé sans cette ligne, le contournement reste désactivé jusqu'à la fin de l'exécution, avec un avertissement dans le journal. Son état est noté dans `provenance.json`. Coût : environ 0,5 s par appel de Siril. À passer à `off` quand Siril sera corrigé.

**D14 — Masque de pixels défectueux par catégories (bits).**
Sur l'IMX294 en 300 s, ≈ 1,8 % des pixels dépassent 5 σ (σ = bruit temporel d'une pose) : les marquer tous rendrait la moitié des mesures suspectes. Le BPM distingue donc *tièdes* (information), *très chauds* (> 50 σ), *instables* (bruit temporel > 5 × médiane, rayons cosmiques exclus), *froids*, *morts* (flat) et *saturés* (dark). La photométrie (M4) ne marque une mesure que pour les bits listés dans `masks.bad_pixel_flags`.

**D15 — Repli « dark mis à l'échelle » (niveau B).**
`calibrate -opt=exp` de Siril attend un dark sans offset : le pipeline soustrait un offset uniforme estimé par la médiane du dark le plus court disponible, puis passe `-bias="=offset"`. Journalisé et visible dans le rapport.

**D16 — Darks de la même nuit uniquement (réponse à Q2, 22/11/2026).**
Constat M2 : les darks de la nuit du 06 au 07/09 (dossier `INTENSITY_light`, 300 s et 180 s) sont **25,7 ADU plus hauts** que ceux des autres nuits, de façon uniforme et sans bruit supplémentaire → saut d'offset électronique, pas une fuite de lumière. Les darks de flat du 07/09 à 13 h sont revenus au niveau habituel (512 ADU). Aucun mot-clé ne l'explique (`OFFSET` absent ; `SWCREATE` change presque chaque nuit sans lien avec le saut).
Règle choisie par l'utilisateur :
- un groupe de lights n'utilise que les darks **pris la même nuit** (niveau A) ; les darks de deux nuits ne sont **jamais combinés** (`matching.dark_same_night: true`) ;
- dans une même nuit, si deux séries ont des niveaux différents (contrôle P1), seules celles du niveau de la série la plus proche des lights sont combinées ;
- sans dark de la même nuit : darks d'**une seule** autre nuit, niveau **B**. Choix fait sur les en-têtes : la nuit la plus proche ; puis, à l'étape des masters (pixels lus), si cette nuit a un offset anormal (P1), la nuit la plus proche **au niveau habituel** est prise à la place. C'est le cas pour sept12 → oct08 : le 06/09 est écarté au profit du 02/09, car l'offset était revenu à la normale dès le 07/09 ;
- ces sessions n'apparaissent pas dans `darks-todo` : on ne peut pas refaire après coup des darks d'une nuit passée.
Effet sur la photométrie : un décalage uniforme est retiré en grande partie par le fond local (anneau). Cette règle sert donc surtout à garder des masters justes, une mise à l'échelle des darks correcte (repli B, qui dépend de l'offset), des contrôles de fond lisibles et une provenance claire.

## Questions ouvertes (à confirmer par l'utilisateur ; défauts entre crochets)

**Q3 — Flats de aout31 pris le 29/08.** Montage inchangé jusqu'au 01/09 ?
[Défaut : oui (les dates de flats d'un même dossier SESSION sont acceptées) ; avertissement dans le rapport.]

**Q4 — Lights du dossier `removed/`.**
[Défaut : ignorés. Option `include_removed` pour les réintégrer : un rejet esthétique, par exemple pour un satellite, n'est pas forcément un rejet photométrique.]

**Q5 — Moteur d'astrométrie.**
[Défaut : solveur de Siril s'il fonctionne hors ligne avec un catalogue local ; sinon ASTAP (base D50 ou D80). À trancher au jalon M3 selon la version de Siril installée.]

**Q6 — Magnitude limite du catalogue Gaia.**
[Défaut : G < 15, à ajuster au jalon M4 selon la profondeur réelle en 300 s avec un filtre étroit.]
