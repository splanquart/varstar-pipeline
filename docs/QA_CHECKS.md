# Contrôles qualité précoces (avant tout calcul de flux)

Objectif : détecter les **erreurs d'acquisition** le plus tôt possible dans le pipeline, avant qu'elles ne produisent des courbes de lumière fausses, et dire à l'observateur **quoi changer à la prochaine séance**.

Principes :
- Un contrôle ne corrige rien en silence : il **signale** (info / avertissement / bloquant), et l'effet sur le niveau de calibration est visible.
- Chaque alerte donne : ce qui a été vu (chiffres), les fichiers concernés, l'impact probable sur la photométrie, et un **conseil pour la prochaine séance**.
- Les contrôles les moins coûteux passent en premier : en-têtes (inventaire, < 1 s), puis statistiques de pixels sur les bruts (quelques secondes), puis contrôles après calibration.
- Section « Contrôles qualité » en tête de chaque rapport (`inventory.md`, `masters.md`, puis `report.html`), et une commande `varstar check <dossier>` qui fait l'inventaire et les contrôles bruts sans rien calculer d'autre, à lancer dès le lendemain d'une séance.

État : ✅ fait · 🟡 partiel · ⬜ à faire.

## 1. En-têtes seulement (étape `inventory`)

| # | Contrôle | Détecte | État |
|---|---|---|---|
| H1 | Dossier incohérent avec l'en-tête (pose, type, filtre) | fichiers mal rangés | ✅ |
| H2 | `OFFSET` absent | impossible de vérifier l'appariement de l'offset | ✅ (avertissement) |
| H3 | `CCD-TEMP` − `SET-TEMP` > tolérance, image par image | refroidissement pas encore stabilisé en début de séquence, consigne non tenue (été) | ✅ |
| H4 | Changement de logiciel ou de pilote (`SWCREATE`, `INSTRUME`) entre lights et calibrations | réglages de caméra réinitialisés par une mise à jour (cas d'offset possible) | ✅ (info) |
| H5 | `FOCPOS` / `FOCTEMP` des flats vs des lights | mise au point refaite entre flats et lights → poussières décalées ; aide à répondre automatiquement aux questions « même montage ? » (Q1, Q3) | ✅ |
| H6 | Pointage des lights (`RA`/`DEC`) loin de la cible ou du centre habituel | mauvais objet, recadrage, cible perdue | ✅ |
| H7 | Météo : `CLOUDCVR` > 0, `AMBTEMP` − `DEWPOINT` < 2 °C | nuages, risque de buée sur l'optique | ✅ |
| H8 | Darks pris télescope pointé vers le ciel (`CENTALT` élevé, nuit claire) | risque de fuite de lumière à vérifier en P2 | ✅ (info) |
| H9 | Nombre de darks / flats insuffisant pour le rejet (< 8 ou < 15) | masters bruités | ✅ (info) |
| H10 | Dark de flat de pose différente de celle des flats | niveau B évitable | ✅ (`darks-todo`) |

## 2. Statistiques de pixels sur les bruts (avant les masters)

| # | Contrôle | Détecte | État |
|---|---|---|---|
| P1 | **Saut de niveau des darks** entre nuits ou séries : écart uniforme, sans bruit supplémentaire | changement d'offset (cas trouvé en 2021 : +25 ADU la nuit du 06/09) ; refuser de combiner deux niveaux dans un même master | ✅ (jamais deux nuits ni deux niveaux dans un master : D16) |
| P2 | **Fuite de lumière** dans les darks : excès non uniforme (gradient, coin), bruit temporel augmenté, corrélé à l'heure ou au pointage | couvercle mal mis, lumière parasite | ✅ (offset ou lumière : la variance ajoutée par des photons vaut Δ/gain ; poses isolées hors série) |
| P3 | Niveau des flats hors 15–85 % de la dynamique ; flats saturés | flats sous- ou surexposés | ✅ |
| P4 | **Forme du master flat comparée à la session voisine** (rapport des deux flats : anneaux de poussière, gradient, rotation) | poussière déplacée, caméra tournée ou démontée → flats non partageables | ✅ (rapports dans `qa/flats/ratio_*.fits` ; blocs de 16 px binés réduits pour garder 4 blocs par côté, contrôle sauté avec un item `info` si l'image binée fait moins de 12 px ; une erreur devient un item `error` sans arrêter le pipeline, [#2](https://github.com/splanquart/varstar-pipeline/issues/2)) |
| P5 | Darks : nombre de pixels chauds et amplitude de l'amp glow comparés à la bibliothèque | dérive du capteur, mauvaise température | ⬜ |
| P6 | Lights : fond de ciel anormal (Lune, aube, nuages), fraction saturée, traînées | images à écarter avant la photométrie | ⬜ (prévu M3/M4 « qualité ») |
| P8 | Médiane des lights − niveau des darks appariés (fond de ciel brut) | darks plus hauts que les lights → offset différent (inopérant si le fond de ciel est fort : ≈ 600–4000 ADU en 2021) | ✅ |
| P9 | **Darks à une autre température** : écart de niveau entre séries de même pose, avec une température CCD moyenne différente de plus de `matching.temp_tolerance_c`, dans le même sens que l'écart de température (testé avant P1/P2) | courant d'obscurité plus fort (refroidissement pas tenu), qui ressemble à une fuite de lumière (excès non uniforme, bruit en plus) ; conseil : vérifier le refroidissement, refaire les darks à la bonne température | ✅ ([#3](https://github.com/splanquart/varstar-pipeline/issues/3)) |
| P7 | Facteur ADU détecté (×4) identique sur toutes les séries | changement de mode de lecture / bits | ✅ (chaque calibration + 3 lights par groupe) |

## 3. Après calibration (avant la photométrie)

| # | Contrôle | Détecte | État |
|---|---|---|---|
| C1 | Résidus de pixels chauds et d'amp glow dans le fond calibré | dark inadapté (température, pose) | ✅ (amp glow : zone comparée au fond voisin, retenu seulement si identique dans tous les filtres) |
| C2 | Anneaux de poussière résiduels dans le fond calibré | flat inadapté | ⬜ |
| C3 | Pixels à 0, NaN, valeurs négatives en masse | dark saturé, erreur de calibration | ✅ |
| C4 | Comparaison statistique avec les masters PixInsight | erreur de calibration | ✅ |

## Mise en œuvre

- Module `src/varstar/qa.py` ; paramètres dans la section `qa:` de la configuration.
- `varstar check <dossier>` : en-têtes + pixels bruts (statistiques sous-échantillonnées mises en cache par fichier dans `qa/frame_stats.json`, ≈ 20 s pour 740 images la première fois). `--headers-only` ne lit aucun pixel. Code de sortie 3 s'il y a une erreur.
- `varstar run` : étape `check` entre `inventory` et `masters` ; contrôles C1/C3 après `calibrate`.
- Rapports : `qa/qa.md` (détail + tableaux : niveau des darks par nuit, fond brut, flats successifs, images calibrées) ; sections « Contrôles qualité » en tête de `inventory.md` (en-têtes) et `masters.md` (en-têtes + bruts).
- Les niveaux sont des moyennes écrêtées (5 σ) et non des médianes : les valeurs brutes sont des multiples de 4 ADU, ce qui quantifie les médianes.

## Cas réel à l'origine de ces notes (NGC 1499, 2021)

- **Saut d'offset dans les darks.** Darks 300 s et 180 s de la nuit du 06/09 : +25,3 ADU uniformes sur tout le capteur, bruit inchangé → offset électronique, pas fuite de lumière. Nuits du 31/08 et du 02/09 : 512–516 ADU ; darks de flat du 07/09 à 13 h : 512 ADU. Aucun mot-clé ne l'explique : `OFFSET` absent, `SWCREATE` change presque chaque nuit (N.I.N.A. 1.11 builds 1133 → 1157) sans lien avec le saut. Conseil pour la prochaine séance : vérifier que N.I.N.A. écrit bien `OFFSET` (pilote natif ZWO plutôt qu'ASCOM), fixer l'offset dans le profil, et faire quelques darks la même nuit que les lights.
- Seul un contrôle sur les pixels (P1) pouvait le voir, et il l'a vu avant la photométrie : c'est le modèle à suivre.
