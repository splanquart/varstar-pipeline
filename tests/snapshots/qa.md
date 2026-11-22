# Contrôles qualité

- Généré le 2026-11-22T10:00:00
- Bilan : 1 erreur(s), 1 avertissement(s), 1 info(s)

## 1. En-têtes

- **H9 — info** : sept06 / H : 5 darks seulement (< 15) pour le master
  - Impact : master dark plus bruité
  - Conseil pour la prochaine séance : faire au moins 15 darks par pose et par nuit

## 2. Pixels des images brutes

- **P1 — avertissement** : darks 300 s, nuit(s) du 06/09 (sept06) : niveau 541.0 ADU contre 516.0 ADU
  - Impact : décalage du fond
  - Conseil pour la prochaine séance : fixer l'offset
  - Fichiers : `2021-09-07_02-00-00_Dark.fits`, `2021-09-07_02-01-00_Dark.fits`, `2021-09-07_02-02-00_Dark.fits` (+2)

## 3. Après calibration

- **C3 — erreur** : sept06 / H : fond calibré négatif sur 1/3 image(s)
  - Fichiers : `a.fits`

## Niveau des darks par nuit (contrôle P1)

| Série | Pose | Nuit | N | Niveau ADU | σ ADU | T °C | Dossier(s) |
|---|---|---|---|---|---|---|---|
| DARK-300s-20210902 | 300 s | 2021-09-02 | 5 | 516.0 | 12.00 | -12.0 | dark/a |
| DARK-300s-20210907 | 300 s | 2021-09-06 | 5 | 541.0 | 12.00 | — | dark/b |

## Fond de ciel brut estimé (contrôle P8)

Médiane des lights − niveau des darks appariés. Une valeur négative signifie des darks plus hauts que les lights.

| Groupe | Lights ADU | Darks ADU | Fond estimé ADU | Niveau dark |
|---|---|---|---|---|
| sept06 / H | 600.0 | 541.0 | +59.0 | A |

## Flats successifs (contrôle P4)

Rapport de deux flats normalisés. *Structure locale* : anneaux de poussière déplacés ; *forme* : vignetage différent (caméra tournée, tirage, panneau). Les rapports sont dans `qa/flats/ratio_*.fits`.

| Filtre | Flat 1 | Flat 2 | Structure locale % | Forme % | Bruit % | Verdict |
|---|---|---|---|---|---|---|
| H | FLAT-H-1 (sept06) | FLAT-H-2 () | 2.10 | 0.40 | 0.10 | poussières déplacées |

## Images calibrées (contrôles C1, C3)

| Groupe | N | Fond min ADU | Fond médian ADU | σ ADU | Amp glow résiduel ADU (zone la plus forte) | NaN max | Zéros max |
|---|---|---|---|---|---|---|---|
| sept06 / H | 3 | -4.0 | 12.0 | 5.00 | — | 0.000 % | 0.000 % |

