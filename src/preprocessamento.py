"""ETL: diagnóstico, validação, limpeza e engenharia de atributos."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .ingestao import hash_linhas
from .particionamento import COLUNAS_TRACO

ATRIBUTOS_DERIVADOS = [
    "relacao_agua_cimento",      # principal fator da resistência (lei de Abrams)
    "aglomerante_total",         # cimento + escória + cinza volante
    "relacao_agua_aglomerante",  # água / aglomerante total
    "log_idade",                 # ganho de resistência é ~logarítmico no tempo
]


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de atributos (determinística, sem parâmetros ajustados -> sem vazamento)."""
    out = df.copy()
    aglomerante = out["cimento"] + out["escoria"] + out["cinza_volante"]
    out["relacao_agua_cimento"] = out["agua"] / out["cimento"]
    out["aglomerante_total"] = aglomerante
    out["relacao_agua_aglomerante"] = out["agua"] / aglomerante.replace(0, np.nan)
    out["log_idade"] = np.log1p(out["idade"])
    return out


def lista_features(cfg) -> list[str]:
    return list(cfg["dados"]["atributos_originais"]) + ATRIBUTOS_DERIVADOS


def diagnostico(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Resumo por coluna: ausentes, violações de faixa e outliers (regra IQR 1,5)."""
    linhas = []
    for col in cfg["dados"]["atributos_originais"] + [cfg["dados"]["alvo"]]:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = cfg["validacao"]["minimos"][col], cfg["validacao"]["maximos"][col]
        linhas.append({
            "coluna": col,
            "ausentes": int(s.isna().sum()),
            "fora_da_faixa": int(((s < lo) | (s > hi)).sum()),
            "outliers_iqr": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "min": s.min(), "mediana": s.median(), "max": s.max(),
        })
    return pd.DataFrame(linhas)


def validar_esquema(df: pd.DataFrame, cfg):
    esperadas = set(cfg["dados"]["atributos_originais"] + [cfg["dados"]["alvo"]])
    faltando = esperadas - set(df.columns)
    if faltando:
        raise ValueError(f"Lote com esquema inválido. Colunas ausentes: {sorted(faltando)}")


def limpar_lote(novo: pd.DataFrame, existente: pd.DataFrame | None, cfg):
    """Remove linhas inválidas e duplicadas (no lote e em relação ao já incorporado).
    Ausências fora da chave de composição e do alvo são imputadas no Pipeline."""
    validar_esquema(novo, cfg)
    cols = cfg["dados"]["atributos_originais"] + [cfg["dados"]["alvo"]]
    novo = novo[cols].copy()
    rel = {"linhas_recebidas": len(novo)}

    alvo = cfg["dados"]["alvo"]
    sem_alvo = novo[alvo].isna()
    sem_traco = novo[list(COLUNAS_TRACO)].isna().any(axis=1)
    viol = pd.Series(False, index=novo.index)
    for col in cols:
        lo, hi = cfg["validacao"]["minimos"][col], cfg["validacao"]["maximos"][col]
        viol |= novo[col].notna() & ((novo[col] < lo) | (novo[col] > hi))
    rel["removidas_alvo_ausente"] = int(sem_alvo.sum())
    rel["removidas_traco_incompleto"] = int((sem_traco & ~sem_alvo).sum())
    rel["removidas_fora_da_faixa"] = int((viol & ~(sem_alvo | sem_traco)).sum())
    novo = novo[~(viol | sem_alvo | sem_traco)]

    dup_interno = novo.duplicated()
    rel["removidas_duplicadas_no_lote"] = int(dup_interno.sum())
    novo = novo[~dup_interno]

    if existente is not None and len(existente):
        ja = set(hash_linhas(existente[cols]))
        dup_ext = hash_linhas(novo).isin(ja).values
        rel["removidas_ja_incorporadas"] = int(dup_ext.sum())
        novo = novo[~dup_ext]
    else:
        rel["removidas_ja_incorporadas"] = 0

    rel["ausentes_para_imputacao"] = int(novo.isna().sum().sum())
    rel["linhas_aceitas"] = len(novo)
    return novo.reset_index(drop=True), rel


def construir_preprocessador(features: list[str]) -> Pipeline:
    """Atributos derivados -> seleção das features eleitas -> imputação -> padronização.
    Ajustado apenas com dados de treino e serializado junto ao modelo."""
    numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])
    return Pipeline([
        ("atributos", FunctionTransformer(criar_atributos)),
        ("selecao", ColumnTransformer([("num", numerico, list(features))], remainder="drop")),
    ])
