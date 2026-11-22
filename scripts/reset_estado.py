"""Zera o estado do pipeline (volta ao ciclo 0). Os lotes e o holdout são mantidos.

Uso: python scripts/reset_estado.py
Atenção: apaga histórico do MLflow, modelos, evidências e treino acumulado.
"""
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ALVOS = [
    "mlflow.db", "mlartifacts", "artifacts", "data/processed/treino_acumulado.csv",
    "data/state/estado_ingestao.json", "models/modelo_vigente.json", "models/historico", "evidencias",
]

for alvo in ALVOS:
    p = RAIZ / alvo
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()
for d in ["models/historico", "evidencias", "data/processed", "data/state"]:
    (RAIZ / d).mkdir(parents=True, exist_ok=True)
    (RAIZ / d / ".gitkeep").touch()
print("Estado zerado.")
