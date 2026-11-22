"""Entrada e versionamento de dados: controla quais lotes já foram incorporados."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from .utils import RAIZ


def carregar_estado(cfg) -> dict:
    arq = RAIZ / cfg["dados"]["estado"]
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    return {"lotes_processados": [], "ciclos": []}


def salvar_estado(cfg, estado: dict):
    arq = RAIZ / cfg["dados"]["estado"]
    arq.parent.mkdir(parents=True, exist_ok=True)
    arq.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")


def listar_lotes(cfg) -> list[Path]:
    return sorted((RAIZ / cfg["dados"]["lotes_dir"]).glob("lote_*.csv"))


def proximo_lote(cfg, estado) -> Path | None:
    """Primeiro lote disponível que ainda não foi incorporado (ordem do nome)."""
    for lote in listar_lotes(cfg):
        if lote.name not in estado["lotes_processados"]:
            return lote
    return None


def carregar_treino_acumulado(cfg) -> pd.DataFrame | None:
    arq = RAIZ / cfg["dados"]["treino_acumulado"]
    return pd.read_csv(arq) if arq.exists() else None


def salvar_treino_acumulado(cfg, df: pd.DataFrame):
    arq = RAIZ / cfg["dados"]["treino_acumulado"]
    arq.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(arq, index=False)


def hash_linhas(df: pd.DataFrame) -> pd.Series:
    """Hash por linha, usado para impedir duplicação ao incorporar novos lotes."""
    return pd.util.hash_pandas_object(df.round(6), index=False)


def versao_dados(df: pd.DataFrame, lotes: list[str]) -> str:
    conteudo = df.round(6).to_csv(index=False).encode()
    return f"v{len(lotes)}-{hashlib.sha256(conteudo).hexdigest()[:8]}"


def carregar_holdout(cfg):
    d = RAIZ / cfg["dados"]["holdout_dir"]
    return pd.read_csv(d / "validacao.csv"), pd.read_csv(d / "teste.csv")
