"""Demonstração com o modelo vigente. Execute com python -m src.prever."""
import argparse
import math

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .promocao import HISTORICO_MODELOS, carregar_manifesto
from .utils import RAIZ, carregar_config

EXEMPLO: dict[str, float] = {
    "cimento": 350, "escoria": 100, "cinza_volante": 0, "agua": 180,
    "superplastificante": 6, "agregado_graudo": 970, "agregado_miudo": 780, "idade": 28,
}


def carregar_modelo_vigente() -> Pipeline:
    manifesto = carregar_manifesto()
    if manifesto is None:
        raise ValueError("Modelo vigente ausente. Execute o pipeline de treinamento primeiro.")
    caminho = (RAIZ / manifesto["arquivo_modelo"]).resolve()
    if not caminho.is_relative_to(HISTORICO_MODELOS.resolve()) or caminho.suffix != ".joblib":
        raise ValueError("O manifesto deve apontar para um modelo em models/historico.")
    if not caminho.is_file():
        raise ValueError("O arquivo do modelo vigente não foi encontrado.")
    return joblib.load(caminho)


def prever(**entradas: float) -> float:
    cfg = carregar_config()
    colunas = cfg["dados"]["atributos_originais"]
    desconhecidas = entradas.keys() - set(colunas)
    if desconhecidas:
        raise ValueError(f"Atributos desconhecidos: {', '.join(sorted(desconhecidas))}.")
    valores = EXEMPLO | entradas
    for coluna in colunas:
        valor = valores[coluna]
        minimo = cfg["validacao"]["minimos"][coluna]
        maximo = cfg["validacao"]["maximos"][coluna]
        if not math.isfinite(valor) or not minimo <= valor <= maximo:
            unidade = "dias" if coluna == "idade" else "kg/m³"
            raise ValueError(f"{coluna}: informe um valor entre {minimo} e {maximo} {unidade}.")
    if not float(valores["idade"]).is_integer():
        raise ValueError("Informe a idade em dias inteiros.")
    quadro = pd.DataFrame([valores], columns=colunas)
    return float(carregar_modelo_vigente().predict(quadro)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Estime a resistência do concreto em MPa.")
    for nome, padrao in EXEMPLO.items():
        unidade = "dias" if nome == "idade" else "kg/m³"
        parser.add_argument(
            f"--{nome.replace('_', '-')}", dest=nome, type=float, default=padrao,
            help=f"{unidade} (padrão: %(default)s)",
        )
    entradas = vars(parser.parse_args())
    try:
        resistencia = prever(**entradas)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Resistência estimada aos {entradas['idade']:g} dias: {resistencia:.2f} MPa")


if __name__ == "__main__":
    main()
