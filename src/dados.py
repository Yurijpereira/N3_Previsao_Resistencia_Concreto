"""Carregamento do dataset e engenharia de atributos.

Dataset: Concrete Compressive Strength (I-Cheng Yeh, 1998)
UCI Machine Learning Repository, id 165 - licença CC BY 4.0
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
"""
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_CSV = RAIZ / "data" / "concrete.csv"

# Nomes em português (unidades: kg por m³ de mistura; idade em dias; resistência em MPa)
COLUNAS = {
    "cement": "cimento",
    "slag": "escoria",
    "ash": "cinza_volante",
    "water": "agua",
    "superplastic": "superplastificante",
    "coarseagg": "agregado_graudo",
    "fineagg": "agregado_miudo",
    "age": "idade",
    "strength": "resistencia",
}
ALVO = "resistencia"
COMPONENTES = ["cimento", "escoria", "cinza_volante", "agua",
               "superplastificante", "agregado_graudo", "agregado_miudo"]
ENTRADAS = COMPONENTES + ["idade"]


def carregar_dados(remover_duplicatas: bool = True) -> pd.DataFrame:
    """Lê o CSV local, traduz as colunas e (por padrão) remove as 25 linhas duplicadas."""
    df = pd.read_csv(CAMINHO_CSV).rename(columns=COLUNAS)
    if remover_duplicatas:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def id_mistura(df: pd.DataFrame) -> NDArray[np.intp]:
    """Identifica cada traço (mesma composição). O mesmo traço aparece em várias idades,
    então usamos esse id para separar treino/teste sem vazamento."""
    return df[COMPONENTES].round(2).astype(str).agg("|".join, axis=1).factorize()[0]


def adicionar_atributos(X: pd.DataFrame) -> pd.DataFrame:
    """Atributos baseados em conhecimento de engenharia civil.

    - relacao_agua_cimento: lei de Abrams, quanto maior, menor a resistência.
    - relacao_agua_aglomerante: considera escória e cinza volante como aglomerantes.
    - log_idade: o ganho de resistência é rápido no início e desacelera com o tempo.
    """
    X = X.copy()
    aglomerante = X["cimento"] + X["escoria"] + X["cinza_volante"]
    X["relacao_agua_cimento"] = X["agua"] / X["cimento"]
    X["relacao_agua_aglomerante"] = X["agua"] / aglomerante
    X["log_idade"] = np.log(X["idade"])
    return X
