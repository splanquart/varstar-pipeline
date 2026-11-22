# 002 — P1/P2 ignorent la température : darks « chauds » pris pour une fuite de lumière

- **Statut** : ouvert
- **Vu le** : 2026-11-22, étape 0 de la refactorisation (minisky, nuit s2)
- **Code** : `src/varstar/qa.py`, `raw_checks` (P1/P2 au niveau des séries) et `classify_excess`

## Symptôme

Dans minisky, les darks de la nuit s2 sont pris avec le capteur à −5 °C (CCD-TEMP et SET-TEMP
à −5) au lieu de −12 °C. Le courant d'obscurité plus fort donne +12,5 ADU et un bruit plus élevé
(shot noise). Diagnostic produit :

> P2 — darks 300 s, nuit(s) du 09/09 (s2) : +12.5 ADU par rapport aux autres nuits (506.0 ADU),
> bruit 9.18 contre 5.97 ADU, excès non uniforme jusqu'à 4.6 ADU dans les coins
> → **fuite de lumière probable**

Avec un modèle sans shot noise, le même excès était classé « saut d'offset électronique ».

## Cause

`classify_excess` ne connaît que deux causes : offset (bruit inchangé) ou lumière (bruit en
shot noise). Un courant d'obscurité plus fort, lié à la température, ressemble exactement à de
la lumière. Or la température est dans les en-têtes, et l'inventaire l'utilise déjà : ces darks
sont écartés de l'appariement (ΔT > `temp_tolerance_c`).

## Impact

Le conseil donné à l'utilisateur est faux (« couvercle opaque, caméra recouverte ») alors que
la vraie action est « vérifier le refroidissement / refaire les darks à −12 °C ». Pas d'impact
sur la calibration elle-même : ces darks ne sont pas utilisés.

## Piste

- Avant de classer un excès, comparer la température CCD moyenne des séries. Si ΔT dépasse
  la tolérance, produire un diagnostic « température différente (ΔT = 7 °C) » avec le conseil
  adapté, plutôt que P1/P2.
- Option : vérifier que l'excès suit l'ordre de grandeur attendu (doublement du courant
  d'obscurité tous les ~6 °C).
- Test : la nuit s2 de minisky doit donner ce nouveau diagnostic. Le test
  `test_check_command` accepte aujourd'hui P1 ou P2 et renvoie à cette issue.
