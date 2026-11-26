#!/usr/bin/env bash
# Couverture des tests par couche (plan d'architecture hexagonale).
#   tools/check_coverage.sh            → mesure + rapport ; code de sortie 1 si un seuil n'est pas atteint
#   ENFORCE=0 tools/check_coverage.sh  → rapport seulement (seuils affichés, non bloquants)
# Seuils : domain ≥ 90 %, application ≥ 85 % (branches comprises). adapters et cli : rapport seulement
# (Siril et le jeu réel sont couverts par les tests e2e).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

"$PY" -m pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=html "$@"

status=0
check() {  # check <couche> <seuil %>
  local layer=$1 threshold=$2
  if [ -z "$(find "src/varstar/$layer" -name '*.py' ! -name '__init__.py' 2>/dev/null)" ]; then
    echo "  $layer : pas encore de code"
    return
  fi
  local total
  total=$("$PY" -m coverage report --include="src/varstar/$layer/*" --format=total 2>/dev/null || echo 0)
  if [ "${total%.*}" -lt "$threshold" ]; then
    echo "  $layer : ${total} % (seuil ${threshold} %) — INSUFFISANT"
    status=1
  else
    echo "  $layer : ${total} % (seuil ${threshold} %)"
  fi
}
echo "Couverture par couche :"
check domain 90
check application 85
echo "  total : $("$PY" -m coverage report --format=total) % (indicatif, sans seuil)"
echo "Rapport détaillé : htmlcov/index.html"
if [ "${ENFORCE:-1}" = "1" ]; then
  exit $status
fi
