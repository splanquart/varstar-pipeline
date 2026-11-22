# 003 — Tailles de blocs (masques, P4) fixes, non adaptées à la taille de l'image

- **Statut** : ouvert
- **Vu le** : 2026-11-22, étape 0 de la refactorisation (minisky, 64×96 px)
- **Code** : `src/varstar/masks.py` (`build_amp_glow`, `build_bpm`), `src/varstar/qa.py` (P4),
  `defaults.yaml` (`masks.amp_glow_block_px: 64`, `masks.flat_local_block_px: 32`,
  `qa.flat_compare_bin: 8`)

## Symptôme

Avec les valeurs par défaut, une image de 64×96 px ne donne qu'un ou deux blocs de 64 px :
la détection d'amp glow ne peut rien trouver (la dispersion entre blocs n'a pas de sens).
Aucune alerte n'est émise : le masque est simplement vide. P4, lui, plante (issue 001).

## Impact

Aucun sur l'ASI294MM en bin 2 (4144×2822). À surveiller avec un autre capteur, un binning
plus fort ou un recadrage (ROI). C'est une dégradation silencieuse, que les règles du projet
interdisent.

## Contournement actuel

La configuration de minisky fixe `amp_glow_block_px: 16`, `flat_local_block_px: 16` et
`flat_compare_bin: 2`.

## Piste

- Vérifier au chargement de la configuration (ou à l'inventaire, avec la taille du capteur)
  qu'il y a assez de blocs : par exemple au moins 4×4. Sinon, avertir et adapter
  (`block = min(block, côté // 4)`), en le journalisant.
- À traiter avec la configuration typée (étape 7) : une validation de `Settings` selon la
  taille du capteur.
