"""Construção, treino e avaliação dos dois algoritmos candidatos."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .features import votar
from .particionamento import grupos_traco
from .preprocessamento import construir_preprocessador, lista_features

ALGORITMOS = {"ridge": Ridge, "random_forest": RandomForestRegressor}


@dataclass(frozen=True)
class Fold:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    features: list[str]


def construir_modelo(algoritmo: str, params: dict, features: list[str], seed: int) -> Pipeline:
    params = dict(params)
    if algoritmo == "random_forest":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
    return Pipeline([
        ("preprocessamento", construir_preprocessador(features)),
        ("modelo", ALGORITMOS[algoritmo](**params)),
    ])


def metricas(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def preparar_folds(treino: pd.DataFrame, cfg: dict) -> list[Fold]:
    """Refaz a votação usando só o treino de cada fold; os candidatos usam os mesmos folds."""
    cv = GroupKFold(n_splits=cfg["avaliacao"]["cv_folds"])
    features = lista_features(cfg)
    folds = []
    for idx_treino, idx_val in cv.split(treino, groups=grupos_traco(treino)):
        treino_fold = treino.iloc[idx_treino]
        _, eleitas, _, _ = votar(treino_fold, features, cfg)
        folds.append(Fold(treino_fold, treino.iloc[idx_val], eleitas))
    return folds


def validacao_cruzada(
    algoritmo: str, params: dict, folds: list[Fold], alvo: str, seed: int,
) -> dict[str, float]:
    resultados = []
    for fold in folds:
        modelo = construir_modelo(algoritmo, params, fold.features, seed)
        modelo.fit(fold.treino.drop(columns=[alvo]), fold.treino[alvo])
        pred = modelo.predict(fold.validacao.drop(columns=[alvo]))
        resultados.append(metricas(fold.validacao[alvo], pred))
    resumo = pd.DataFrame(resultados)
    return {
        **{f"cv_{nome}_media": float(valor) for nome, valor in resumo.mean().items()},
        "cv_mae_desvio": float(resumo["mae"].std(ddof=0)),
    }


def baseline_mae(y_treino, y_val) -> float:
    dummy = DummyRegressor(strategy="mean").fit(np.zeros((len(y_treino), 1)), y_treino)
    return float(mean_absolute_error(y_val, dummy.predict(np.zeros((len(y_val), 1)))))


def treinar_e_avaliar(
    algoritmo: str, params: dict, features: list[str], treino: pd.DataFrame,
    validacao: pd.DataFrame, cfg: dict, folds: list[Fold],
) -> tuple[Pipeline, dict[str, float]]:
    alvo = cfg["dados"]["alvo"]
    seed = cfg["projeto"]["seed"]
    X_tr, y_tr = treino.drop(columns=[alvo]), treino[alvo]
    X_val, y_val = validacao.drop(columns=[alvo]), validacao[alvo]

    modelo = construir_modelo(algoritmo, params, features, seed)
    cv = validacao_cruzada(algoritmo, params, folds, alvo, seed)
    modelo.fit(X_tr, y_tr)
    val = {f"val_{k}": v for k, v in metricas(y_val, modelo.predict(X_val)).items()}
    return modelo, {**cv, **val}
