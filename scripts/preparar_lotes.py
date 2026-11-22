"""Prepara (uma única vez) os dados do projeto a partir do arquivo original da UCI.

1. Renomeia colunas para português.
2. Remove duplicatas exatas ANTES da divisão (evita a mesma amostra em treino e teste).
3. Separa teste (20%) e validação fixa (15%) com seed fixa -> data/holdout/.
4. Divide o restante em 4 lotes que simulam a chegada de novos dados:
   - lote_01: base inicial (aleatória)
   - lote_02, lote_03: lotes aleatórios adicionais
   - lote_04: corpos de prova de longa idade (simula mudança de perfil -> drift)
5. Injeta problemas controlados para demonstrar o ETL:
   - lote_02: 3 valores ausentes em 'superplastificante'
   - lote_03: 5 linhas repetidas do lote_01 + 1 linha inválida (água negativa)

Uso: python scripts/preparar_lotes.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((RAIZ / "config/config.yaml").read_text(encoding="utf-8"))
SEED = CFG["projeto"]["seed"]

MAPA = {
    "cement": "cimento", "slag": "escoria", "ash": "cinza_volante", "water": "agua",
    "superplastic": "superplastificante", "coarseagg": "agregado_graudo",
    "fineagg": "agregado_miudo", "age": "idade", "strength": "resistencia",
}


def main():
    df = pd.read_csv(RAIZ / CFG["dados"]["fonte_original"]).rename(columns=MAPA)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Original: {n0} linhas | após remover duplicatas: {len(df)}")

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_teste = int(round(len(df) * CFG["dados"]["frac_teste"]))
    n_val = int(round(len(df) * CFG["dados"]["frac_validacao"]))
    teste = df.iloc[:n_teste]
    validacao = df.iloc[n_teste:n_teste + n_val]
    resto = df.iloc[n_teste + n_val:].reset_index(drop=True)

    # lote_04: amostras de longa idade (>= 90 dias) -> mudança de perfil dos dados
    lote_04 = resto[resto["idade"] >= 90]
    resto = resto.drop(lote_04.index).reset_index(drop=True)
    lote_02 = resto.iloc[:100].copy()
    lote_03 = resto.iloc[100:200].copy()
    lote_01 = resto.iloc[200:].copy()

    rng = np.random.default_rng(SEED)
    idx_nan = rng.choice(len(lote_02), size=3, replace=False)
    lote_02.iloc[idx_nan, lote_02.columns.get_loc("superplastificante")] = np.nan

    repetidas = lote_01.sample(5, random_state=SEED)
    invalida = lote_03.iloc[[0]].copy()
    invalida["agua"] = -1.0
    lote_03 = pd.concat([lote_03, repetidas, invalida], ignore_index=True)

    holdout = RAIZ / CFG["dados"]["holdout_dir"]
    lotes = RAIZ / CFG["dados"]["lotes_dir"]
    holdout.mkdir(parents=True, exist_ok=True)
    lotes.mkdir(parents=True, exist_ok=True)
    teste.to_csv(holdout / "teste.csv", index=False)
    validacao.to_csv(holdout / "validacao.csv", index=False)
    for i, lote in enumerate([lote_01, lote_02, lote_03, lote_04], start=1):
        lote.to_csv(lotes / f"lote_{i:02d}.csv", index=False)
        print(f"lote_{i:02d}: {len(lote)} linhas")
    print(f"validacao: {len(validacao)} | teste: {len(teste)}")


if __name__ == "__main__":
    main()
