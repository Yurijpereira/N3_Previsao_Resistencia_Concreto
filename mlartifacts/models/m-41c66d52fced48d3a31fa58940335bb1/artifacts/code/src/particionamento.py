"""Separa ensaios por composição, mantendo juntas as diferentes idades."""
from itertools import combinations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

COLUNAS_TRACO = (
    "cimento", "escoria", "cinza_volante", "agua", "agregado_graudo", "agregado_miudo",
)


def grupos_traco(df: pd.DataFrame) -> pd.Series:
    """Agrupa pelos ingredientes sempre observados, antes de qualquer imputação.

    Ignorar o superplastificante mantém juntos os ensaios de composições
    semelhantes mesmo quando esse valor está ausente em um dos ensaios.
    """
    ingredientes = df[list(COLUNAS_TRACO)].round(2)
    if ingredientes.isna().any().any():
        raise ValueError("Uma composição está incompleta e não pode ser agrupada por traço.")
    return pd.util.hash_pandas_object(ingredientes.astype("float64"), index=False)


def validar_separacao(**particoes: pd.DataFrame) -> None:
    grupos = {nome: set(grupos_traco(df)) for nome, df in particoes.items()}
    for a, b in combinations(grupos, 2):
        if grupos[a] & grupos[b]:
            raise ValueError(f"Há traços em comum entre {a} e {b}. Confira o particionamento.")


def separar(
    df: pd.DataFrame, frac_teste: float, frac_validacao: float, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grupos = grupos_traco(df)
    teste = GroupShuffleSplit(n_splits=1, test_size=frac_teste, random_state=seed)
    resto_idx, teste_idx = next(teste.split(df, groups=grupos))
    resto = df.iloc[resto_idx]
    validacao = GroupShuffleSplit(
        n_splits=1, test_size=frac_validacao / (1 - frac_teste), random_state=seed + 1,
    )
    treino_idx, val_idx = next(validacao.split(resto, groups=grupos.iloc[resto_idx]))
    treino, val, teste = resto.iloc[treino_idx], resto.iloc[val_idx], df.iloc[teste_idx]
    validar_separacao(treino=treino, validacao=val, teste=teste)
    return treino, val, teste
