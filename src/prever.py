"""Previsão para uma nova mistura de concreto.

Uso em Python:
    from src.prever import prever
    prever(cimento=350, escoria=0, cinza_volante=0, agua=180, superplastificante=5,
           agregado_graudo=1000, agregado_miudo=800, idade=28)

Uso na linha de comando:
    python -m src.prever --cimento 350 --agua 180 --idade 28
"""
import argparse
import math
from functools import lru_cache
from numbers import Real
from threading import Lock
from typing import TypedDict

import joblib
import pandas as pd

from src.treinar import CAMINHO_MODELO, PacoteModelo
from src.treinar import main as treinar

# Valores padrão: um traço comum, com base nas medianas do dataset
PADRAO: dict[str, float] = {"cimento": 275.0, "escoria": 20.0, "cinza_volante": 0.0, "agua": 185.0,
          "superplastificante": 6.0, "agregado_graudo": 968.0, "agregado_miudo": 780.0,
          "idade": 28}

_TRAVA_MODELO = Lock()


class Previsao(TypedDict):
    resistencia_MPa: float
    intervalo_80: tuple[float, float]
    avisos: list[str]


@lru_cache(maxsize=1)
def _carregar_modelo() -> PacoteModelo:
    if not CAMINHO_MODELO.exists():
        print("Modelo não encontrado. Treinando agora…")
        treinar()
    return joblib.load(CAMINHO_MODELO)


def carregar_modelo() -> PacoteModelo:
    """Carrega o modelo local e serializa o primeiro acesso no processo."""
    with _TRAVA_MODELO:
        return _carregar_modelo()


def prever(**entradas: float) -> Previsao:
    """Retorna a resistência em MPa, a faixa empírica e os avisos de extrapolação."""
    desconhecidas = entradas.keys() - PADRAO.keys()
    if desconhecidas:
        raise ValueError(f"Entradas desconhecidas: {', '.join(sorted(desconhecidas))}.")

    valores = {**PADRAO, **entradas}
    for coluna, valor in valores.items():
        if isinstance(valor, bool) or not isinstance(valor, Real) or not math.isfinite(valor):
            raise ValueError(f"{coluna}: informe um número finito.")
        if valor < 0 or (coluna in ("cimento", "agua", "idade") and valor == 0):
            raise ValueError(f"{coluna}: quantidade ou idade fora do domínio da previsão.")
    if not float(valores["idade"]).is_integer():
        raise ValueError("idade: informe o número inteiro de dias.")

    pacote = carregar_modelo()
    X = pd.DataFrame([valores])[pacote["entradas"]]

    avisos = []
    for col, (mn, mx) in pacote["faixas"].items():
        v = valores[col]
        if v < mn or v > mx:
            avisos.append(f"{col} = {v} está fora da faixa vista no treino ({mn:g} a {mx:g}); "
                          "a previsão pode não ser confiável.")

    p = float(pacote["modelo"].predict(X)[0])
    m = pacote["margem_80"]
    return {"resistencia_MPa": round(p, 1),
            "intervalo_80": (round(max(p - m, 0), 1), round(p + m, 1)),
            "avisos": avisos}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prevê a resistência à compressão do concreto (MPa).")
    for nome, v in PADRAO.items():
        ap.add_argument(f"--{nome.replace('_', '-')}", dest=nome, type=float, default=v)
    try:
        r = prever(**vars(ap.parse_args()))
    except ValueError as erro:
        ap.error(str(erro))
    print(f"Resistência estimada: {r['resistencia_MPa']} MPa "
          f"(faixa empírica de referência: {r['intervalo_80'][0]} a {r['intervalo_80'][1]} MPa)")
    for a in r["avisos"]:
        print("Aviso:", a)


if __name__ == "__main__":
    main()
