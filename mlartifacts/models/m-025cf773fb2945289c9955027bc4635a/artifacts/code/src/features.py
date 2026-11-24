"""Três métodos de seleção/importância de atributos e votação."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .preprocessamento import criar_atributos


def _matriz(df_treino, features, alvo):
    """Imputação e padronização ajustadas SOMENTE no treino."""
    X = criar_atributos(df_treino)[features]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features)
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)
    return X, Xs, df_treino[alvo].values


def _ranking(valores: pd.Series, maior_melhor=True) -> pd.Series:
    return valores.rank(ascending=not maior_melhor, method="min").astype(int)


def metodo_selectkbest(X, y, k):
    sel = SelectKBest(f_regression, k=k).fit(X, y)
    score = pd.Series(sel.scores_, index=X.columns)
    return score, _ranking(score), set(X.columns[sel.get_support()])


def metodo_random_forest(X, y, k, seed):
    rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    rank = _ranking(imp)
    return imp, rank, set(imp.sort_values(ascending=False).index[:k])


def metodo_rfe(Xs, y, k):
    rfe = RFE(Ridge(alpha=1.0), n_features_to_select=k, step=1).fit(Xs, y)
    rank = pd.Series(rfe.ranking_, index=Xs.columns)  # 1 = selecionada
    rank_completo = rank.rank(method="first").astype(int)  # ordem de eliminação
    return rank, rank_completo, set(Xs.columns[rfe.support_])


def votar(df_treino: pd.DataFrame, features: list[str], cfg):
    """Executa os três métodos e aplica a regra de votação definida no config."""
    alvo = cfg["dados"]["alvo"]
    k = cfg["votacao"]["top_k"]
    seed = cfg["projeto"]["seed"]
    X, Xs, y = _matriz(df_treino, features, alvo)

    s_kb, r_kb, v_kb = metodo_selectkbest(X, y, k)
    s_rf, r_rf, v_rf = metodo_random_forest(X, y, k, seed)
    s_rfe, r_rfe, v_rfe = metodo_rfe(Xs, y, k)

    tabela = pd.DataFrame({
        "feature": features,
        "score_f_regression": s_kb.values,
        "importancia_rf": s_rf.values,
        "ranking_rfe_sklearn": s_rfe.values,
        "rank_kbest": r_kb.values,
        "rank_rf": r_rf.values,
        "rank_rfe": r_rfe.values,
        "voto_kbest": [int(f in v_kb) for f in features],
        "voto_rf": [int(f in v_rf) for f in features],
        "voto_rfe": [int(f in v_rfe) for f in features],
    })
    tabela["votos"] = tabela[["voto_kbest", "voto_rf", "voto_rfe"]].sum(axis=1)
    tabela["ranking_medio"] = tabela[["rank_kbest", "rank_rf", "rank_rfe"]].mean(axis=1)
    tabela = tabela.sort_values(["votos", "ranking_medio"], ascending=[False, True]).reset_index(drop=True)

    eleitas = tabela.loc[tabela["votos"] >= cfg["votacao"]["votos_minimos"], "feature"].tolist()
    criterio = f">= {cfg['votacao']['votos_minimos']} votos"
    if len(eleitas) < cfg["votacao"]["min_features"]:
        faltam = cfg["votacao"]["min_features"] - len(eleitas)
        extras = tabela.loc[~tabela["feature"].isin(eleitas), "feature"].head(faltam).tolist()
        eleitas += extras
        criterio += f" + desempate por ranking médio ({', '.join(extras)})"
    tabela["eleita"] = tabela["feature"].isin(eleitas).astype(int)

    metodos = {
        "selectkbest_f_regression": sorted(v_kb),
        "importancia_random_forest": sorted(v_rf),
        "rfe_ridge": sorted(v_rfe),
    }
    return tabela, eleitas, metodos, criterio
