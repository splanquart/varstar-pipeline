# 001 — P4 plante sur les images de moins de 128 px

- **Statut** : ouvert
- **Vu le** : 2026-11-22, étape 0 de la refactorisation (jeu e2e « minisky », 64×96 px)
- **Code** : `src/varstar/qa.py`, `flat_ratio_metrics` → `_block_median`

## Symptôme

`varstar run` / `varstar check` s'arrêtent avec :

```
ValueError: cannot reshape array of size 0 into shape (0,0,newaxis)
```

## Cause

P4 (comparaison des flats successifs) bine les flats par `qa.flat_compare_bin` (8 par défaut),
puis calcule une médiane par blocs de 16 sur l'image binée. Il faut donc au moins
8 × 16 = 128 px dans chaque dimension ; en dessous, `_block_median` reçoit un tableau vide.

## Impact

Aucun sur les vraies images (4144×2822). Mais une exception non rattrapée dans un contrôle
qualité arrête tout le pipeline, ce qui est contraire à l'esprit de la QA (signaler, pas bloquer).

## Contournement actuel

La configuration de minisky fixe `qa.flat_compare_bin: 2`.

## Piste

- Ramener `block` à `min(block, h // 2, w // 2)` ou sauter P4 avec un item `info` (« image trop
  petite pour comparer les flats ») quand la taille ne suffit pas.
- Plus largement, un contrôle qui échoue devrait produire un item QA `error` plutôt que
  d'interrompre l'étape.
- Ajouter un test de domaine sur `flat_ratio_metrics` avec de petites tailles (candidat
  hypothesis : toute forme ≥ 1×1 → pas d'exception).
