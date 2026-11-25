"""Definição dos modelos e da estratégia de avaliação."""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.dados import adicionar_atributos

SEMENTE = 42
Regressor = Pipeline | DummyRegressor


def criar_modelos() -> dict[str, Regressor]:
    """O pré-processamento dos modelos que o utilizam acompanha o estimador."""
    atributos = FunctionTransformer(adicionar_atributos)
    return {
        "Baseline (média)": DummyRegressor(strategy="mean"),
        "Regressão linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Regressão linear + atributos": make_pipeline(atributos, StandardScaler(), LinearRegression()),
        "Random Forest": make_pipeline(
            atributos,
            RandomForestRegressor(n_estimators=400, min_samples_leaf=1,
                                  random_state=SEMENTE, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(
            atributos,
            GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                                      subsample=0.8, random_state=SEMENTE)),
    }


def separar_treino_teste(
    X: pd.DataFrame, y: pd.Series, grupos: NDArray[np.intp], tamanho_teste: float = 0.2,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Separa por TRAÇO: todas as idades de uma mesma mistura ficam do mesmo lado.
    Assim o teste simula o uso real: prever a resistência de uma mistura nunca vista."""
    gss = GroupShuffleSplit(n_splits=1, test_size=tamanho_teste, random_state=SEMENTE)
    idx_treino, idx_teste = next(gss.split(X, y, grupos))
    return idx_treino, idx_teste


METRICAS_CV = {"MAE": "neg_mean_absolute_error",
               "RMSE": "neg_root_mean_squared_error",
               "R2": "r2"}


def validacao_cruzada(
    modelo: Regressor, X: pd.DataFrame, y: pd.Series,
    grupos: NDArray[np.intp] | None = None, n_folds: int = 5,
) -> dict[str, float]:
    """GroupKFold (por traço) quando grupos é informado; KFold aleatório caso contrário."""
    if grupos is not None:
        cv, kw = GroupKFold(n_splits=n_folds), {"groups": grupos}
    else:
        cv, kw = KFold(n_splits=n_folds, shuffle=True, random_state=SEMENTE), {}
    r = cross_validate(modelo, X, y, cv=cv, scoring=METRICAS_CV, **kw)
    saida = {}
    for nome in METRICAS_CV:
        v = r[f"test_{nome}"]
        v = -v if nome != "R2" else v
        saida[nome] = float(np.mean(v))
        saida[f"{nome}_dp"] = float(np.std(v))
    return saida


def metricas(y_real: pd.Series, y_prev: NDArray[np.floating]) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_real, y_prev)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_prev))),
        "R2": float(r2_score(y_real, y_prev)),
    }


def faixa_idade(idade: pd.Series) -> pd.Series:
    return pd.cut(idade, bins=[0, 7, 28, 90, 365],
                  labels=["1-7 dias", "8-28 dias", "29-90 dias", "91-365 dias"])
