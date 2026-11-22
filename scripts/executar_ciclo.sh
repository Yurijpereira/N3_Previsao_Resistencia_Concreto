#!/usr/bin/env bash
# Executa um ciclo completo localmente, na mesma ordem do workflow do GitHub Actions.
# Uso: bash scripts/executar_ciclo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p artifacts/notebooks_executados
for nb in 01_etl 02_votacao_features 03_treino_competicao 04_selecao_promocao 05_previsoes; do
  echo ">> executando $nb"
  jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
    --ExecutePreprocessor.timeout=900 \
    --output-dir artifacts/notebooks_executados --output "${nb}_executado.ipynb"
done
python -m src.finalizar_ciclo
