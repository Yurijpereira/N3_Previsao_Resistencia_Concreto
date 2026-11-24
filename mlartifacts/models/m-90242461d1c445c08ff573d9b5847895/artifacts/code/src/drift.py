"""Recurso adicional (critério 10): monitoramento de drift dos dados de entrada.

Problema: os lotes novos podem ter um perfil diferente do que o modelo vigente conheceu.
Solução: a cada ciclo, compara a distribuição de cada variável do lote novo com o treino
acumulado (teste de Kolmogorov-Smirnov). Gera tabela, gráfico, métricas no MLflow e um
alerta visível no resumo da execução do GitHub Actions.
"""
import pandas as pd
from scipy.stats import ks_2samp


def detectar_drift(referencia: pd.DataFrame, novo: pd.DataFrame, colunas: list[str], cfg):
    alpha = cfg["drift"]["alpha"]
    linhas = []
    for col in colunas:
        a, b = referencia[col].dropna(), novo[col].dropna()
        est, p = ks_2samp(a, b)
        linhas.append({
            "variavel": col, "ks_estatistica": round(float(est), 4), "p_valor": float(p),
            "media_referencia": round(float(a.mean()), 3), "media_lote": round(float(b.mean()), 3),
            "drift": bool(p < alpha),
        })
    tabela = pd.DataFrame(linhas).sort_values("ks_estatistica", ascending=False).reset_index(drop=True)
    frac = float(tabela["drift"].mean())
    alerta = frac >= cfg["drift"]["fracao_alerta"]
    return tabela, frac, alerta
