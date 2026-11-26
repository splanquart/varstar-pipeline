# Masters et masques

- Bibliothèque : `/lib`
- Siril 1.4.5 — généré le 2026-11-26T09:00:00

## Contrôles qualité

_0 erreur(s), 0 avertissement(s), 1 info(s) — détail et méthode : `qa/qa.md`, docs/QA_CHECKS.md._

- **H9 — info** : s1 / H : 8 darks seulement (< 15) pour le master

## Masters

| Type | Master | N | Rejet | Médiane ADU | σ 1 pose ADU | Pixels instables | Niveau flats (% dyn.) |
|---|---|---|---|---|---|---|---|
| dark | `DARK_g120_ox_b2_300s_T-12` (d1) | 8 | sigma clipping | 516.4 | 12.31 | 42 | — |
| dark de flat | `DARKFLAT_g120_ox_b2_1.13s` (df1) | 3 | rejet par percentile | 500.2 | — | — | — |
| flat | `FLAT_g120_ox_b2_1.13s_H_20210907` (f1) | 5 | rejet par percentile | 30000.0 | — | — | 45–47 |

## Groupes de lights

| Groupe | Niveau | Dark | Flat | Dark de flat | Masque BPM | Amp glow |
|---|---|---|---|---|---|---|
| s1 / H (300 s) | **A** | 8 × 300 s | 5 flats | 1.13 s | tièdes 120, très chauds 3, morts (flat) 1 | 4.1 % du champ (excès max 38 ADU) |
| 2021-09-10 / S (300 s) | **B** | 5 × 180 s (mis à l'échelle) | aucun | — | — | — |

Masque BPM : *tièdes* = excès de dark > k·σ d'une pose (bien corrigés par le dark, information seulement) ; *très chauds*, *instables*, *froids*, *morts*, *saturés* = pixels dont la correction est douteuse. Les pixels ne sont jamais modifiés : les mesures touchées seront marquées (jalon M4).

## Validation croisée avec les masters PixInsight

| Master | PixInsight | Résultat |
|---|---|---|
| `DARK_g120_ox_b2_300s_T-12` | `masterDark_EXPOSURE-300.00s.xisf` | médianes 516.4 / 515.9 ADU, différence +0.50 ADU, dispersion 1.20 ADU — **image PixInsight retournée verticalement** |
| `FLAT_g120_ox_b2_1.13s_H_20210907` | `masterFlat_FILTER-H.xisf` | comparaison impossible : image XISF compressée |

## Cohérence des darks par dossier (question Q2)

### `DARK_g120_ox_b2_300s_T-12`

| Dossier | N | Niveau médian ADU | σ ADU | T moy. °C | Excès coins (HG/HD/BG/BD) ADU |
|---|---|---|---|---|---|
| dark/a | 4 | 516.0 | 12.00 | -12.0 | 0.4 / 0.1 / 6.2 / 0.0 |
| dark/b | 4 | 519.5 | 12.40 | -11.9 | 0.4 / 0.1 / 6.2 / 0.0 |

Écart des niveaux médians : 3.5 ADU → différence significative — à examiner.

## Avertissements

- 2021-09-10_S_300s : dark mis à l'échelle depuis 180 s (offset estimé 500.0 ADU) — niveau B
