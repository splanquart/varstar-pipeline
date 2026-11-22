#!/usr/bin/env bash
# Couverture des tests par couche (plan d'architecture hexagonale).
#   tools/check_coverage.sh            → mesure + rapport (seuils affichés, non bloquants)
#   ENFORCE=1 tools/check_coverage.sh  → seuils bloquants (activés à la fin de la migration, étape 8)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

"$PY" -m pytest -q --cov --cov-report=term-missing:skip-covered --cov-report=html "$@"

status=0
check() {  # check <couche> <seuil %>
  local layer=$1 threshold=$2
  if [ ! -d "src/varstar/$layer" ]; then
    echo "  $layer : couche pas encore créée"
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
if [ "${ENFORCE:-0}" = "1" ]; then
  exit $status
fi
