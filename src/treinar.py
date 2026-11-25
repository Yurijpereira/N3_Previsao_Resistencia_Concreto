"""Treina e avalia os modelos e salva o modelo final usado no app.

Uso (na raiz do projeto):
    python -m src.treinar
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict

from src.dados import ALVO, ENTRADAS, RAIZ, carregar_dados, id_mistura
from src.modelos import (
    SEMENTE,
    Regressor,
    criar_modelos,
    faixa_idade,
    metricas,
    separar_treino_teste,
    validacao_cruzada,
)

PASTA_MODELOS = RAIZ / "models"
PASTA_RELATORIOS = RAIZ / "reports"
CAMINHO_MODELO = PASTA_MODELOS / "modelo.joblib"


class PacoteModelo(TypedDict):
    nome: str
    modelo: Regressor
    margem_80: float
    entradas: list[str]
    faixas: dict[str, tuple[float, float]]
    metricas_teste: dict[str, float]
    cobertura_intervalo: float


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    PASTA_MODELOS.mkdir(exist_ok=True)
    PASTA_RELATORIOS.mkdir(exist_ok=True)

    df = carregar_dados()
    X, y, grupos = df[ENTRADAS], df[ALVO], id_mistura(df)
    print(f"{len(df)} amostras (após remover duplicatas) | {grupos.max() + 1} traços distintos")

    idx_tr, idx_te = separar_treino_teste(X, y, grupos)
    X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
    g_tr = grupos[idx_tr]
    if np.intersect1d(g_tr, grupos[idx_te]).size:
        raise RuntimeError("O mesmo traço apareceu no treino e no teste.")
    print(f"Treino: {len(X_tr)} | Teste: {len(X_te)} (traços do teste nunca aparecem no treino)\n")

    # 1) Validação cruzada no treino: por traço (honesta) e aleatória (otimista)
    modelos = criar_modelos()
    linhas: list[dict[str, str | float]] = []
    for nome, modelo in modelos.items():
        cv_grupo = validacao_cruzada(modelo, X_tr, y_tr, grupos=g_tr)
        cv_aleat = validacao_cruzada(modelo, X_tr, y_tr, grupos=None)
        linhas.append({"modelo": nome,
                       **{f"cv_{k}": v for k, v in cv_grupo.items()},
                       "cv_aleatorio_MAE": cv_aleat["MAE"], "cv_aleatorio_R2": cv_aleat["R2"]})
        print(f"{nome:<30} MAE {cv_grupo['MAE']:5.2f} ± {cv_grupo['MAE_dp']:.2f} MPa | "
              f"R² {cv_grupo['R2']:.3f} | (CV aleatória: MAE {cv_aleat['MAE']:.2f})")
    tabela_cv = pd.DataFrame(linhas)

    # 2) Escolha do melhor pelo MAE da validação por traço e avaliação no teste
    melhor_nome = str(tabela_cv.loc[tabela_cv["cv_MAE"].idxmin(), "modelo"])
    print(f"\nMelhor modelo na validação cruzada: {melhor_nome}")

    resultados_teste = {}
    melhor_ajustado = clone(modelos[melhor_nome]).fit(X_tr, y_tr)
    previsoes = pd.DataFrame({"real": y_te.values, "idade": X_te["idade"].values})
    for nome, modelo in modelos.items():
        m = melhor_ajustado if nome == melhor_nome else clone(modelo).fit(X_tr, y_tr)
        p = m.predict(X_te)
        resultados_teste[nome] = metricas(y_te, p)
        previsoes[nome] = p
    tabela_cv["teste_MAE"] = tabela_cv["modelo"].map(lambda n: resultados_teste[n]["MAE"])
    tabela_cv["teste_RMSE"] = tabela_cv["modelo"].map(lambda n: resultados_teste[n]["RMSE"])
    tabela_cv["teste_R2"] = tabela_cv["modelo"].map(lambda n: resultados_teste[n]["R2"])
    print("Teste:", {k: round(v, 3) for k, v in resultados_teste[melhor_nome].items()})

    # 3) Análise de erros do melhor modelo
    previsoes["erro"] = previsoes[melhor_nome] - previsoes["real"]
    previsoes["faixa_idade"] = faixa_idade(previsoes["idade"])
    erro_por_idade = (previsoes.assign(abs_erro=previsoes["erro"].abs())
                      .groupby("faixa_idade", observed=True)["abs_erro"]
                      .agg(["count", "mean"]).rename(columns={"count": "n", "mean": "MAE"}))
    print("\nMAE por faixa de idade:\n", erro_por_idade.round(2))

    # 4) Intervalo de previsão de 80%: margem = percentil 80 do erro absoluto
    #    nas previsões fora da amostra (validação cruzada por traço no treino).
    oof = cross_val_predict(modelos[melhor_nome], X_tr, y_tr,
                            cv=GroupKFold(n_splits=5), groups=g_tr)
    margem = float(np.quantile(np.abs(oof - y_tr.values), 0.80))
    p_te = previsoes[melhor_nome].values
    cobertura = float((np.abs(p_te - y_te.values) <= margem).mean())
    print(f"\nIntervalo 80%: previsão ± {margem:.1f} MPa | cobertura no teste = {cobertura:.1%}")

    # 5) Importância por permutação (no teste)
    imp = permutation_importance(melhor_ajustado, X_te, y_te, n_repeats=20,
                                 random_state=SEMENTE, scoring="neg_mean_absolute_error")
    importancia = (pd.Series(imp.importances_mean, index=ENTRADAS)
                   .sort_values(ascending=False).rename("aumento_MAE"))

    # 6) Modelo final: retreina com todos os dados para o app
    final = clone(modelos[melhor_nome]).fit(X, y)
    pacote: PacoteModelo = {
        "nome": melhor_nome,
        "modelo": final, "margem_80": margem,
        "entradas": ENTRADAS,
        "faixas": {c: (float(X[c].min()), float(X[c].max())) for c in ENTRADAS},
        "metricas_teste": resultados_teste[melhor_nome],
        "cobertura_intervalo": cobertura,
    }
    # 7) Relatórios
    tabela_cv.round(4).to_csv(PASTA_RELATORIOS / "comparacao_modelos.csv", index=False)
    previsoes.round(3).to_csv(PASTA_RELATORIOS / "previsoes_teste.csv", index=False)
    with open(PASTA_RELATORIOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump({"melhor_modelo": melhor_nome,
                   "teste": {
                       nome: {metrica: round(valor, 6) for metrica, valor in resultado.items()}
                       for nome, resultado in resultados_teste.items()
                   },
                   "erro_por_idade": erro_por_idade.round(3).reset_index().astype({"faixa_idade": str}).to_dict("records"),
                   "intervalo_80": {"margem_MPa": margem, "cobertura_teste": cobertura},
                   "importancia_permutacao": importancia.round(3).to_dict()},
                  f, ensure_ascii=False, indent=2)
    with TemporaryDirectory(dir=PASTA_MODELOS) as temporario:
        arquivo_temporario = Path(temporario) / CAMINHO_MODELO.name
        joblib.dump(pacote, arquivo_temporario, compress=3)
        arquivo_temporario.replace(CAMINHO_MODELO)
    print(f"\nModelo salvo em {CAMINHO_MODELO.relative_to(RAIZ)}")
    return tabela_cv, previsoes, importancia


if __name__ == "__main__":
    main()
