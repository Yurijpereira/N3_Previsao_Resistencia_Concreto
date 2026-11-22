"""Construção, treino e avaliação dos dois algoritmos candidatos."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from .preprocessamento import construir_preprocessador

ALGORITMOS = {"ridge": Ridge, "random_forest": RandomForestRegressor}


def construir_modelo(algoritmo: str, params: dict, features: list[str], seed: int) -> Pipeline:
    params = dict(params)
    if algoritmo == "random_forest":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
    return Pipeline([
        ("preprocessamento", construir_preprocessador(features)),
        ("modelo", ALGORITMOS[algoritmo](**params)),
    ])


def metricas(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def validacao_cruzada(modelo, X, y, folds, seed) -> dict:
    """CV no treino; o pré-processamento é reajustado dentro de cada fold (Pipeline)."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(modelo, X, y, cv=cv, scoring={
        "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"})
    return {
        "cv_mae_media": float(-res["test_mae"].mean()), "cv_mae_desvio": float(res["test_mae"].std()),
        "cv_rmse_media": float(-res["test_rmse"].mean()), "cv_r2_media": float(res["test_r2"].mean()),
    }


def baseline_mae(y_treino, y_val) -> float:
    dummy = DummyRegressor(strategy="mean").fit(np.zeros((len(y_treino), 1)), y_treino)
    return float(mean_absolute_error(y_val, dummy.predict(np.zeros((len(y_val), 1)))))


def treinar_e_avaliar(algoritmo, params, features, treino: pd.DataFrame, validacao: pd.DataFrame, cfg):
    alvo = cfg["dados"]["alvo"]
    seed = cfg["projeto"]["seed"]
    X_tr, y_tr = treino.drop(columns=[alvo]), treino[alvo]
    X_val, y_val = validacao.drop(columns=[alvo]), validacao[alvo]

    modelo = construir_modelo(algoritmo, params, features, seed)
    cv = validacao_cruzada(modelo, X_tr, y_tr, cfg["avaliacao"]["cv_folds"], seed)
    modelo.fit(X_tr, y_tr)
    val = {f"val_{k}": v for k, v in metricas(y_val, modelo.predict(X_val)).items()}
    return modelo, {**cv, **val}
