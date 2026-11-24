"""Utilidades compartilhadas: configuração, caminhos, contexto do ciclo e MLflow."""
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
DIR_CICLO = RAIZ / "artifacts" / "ciclo_atual"   # área temporária do ciclo (não versionada)
ARQ_CONTEXTO = DIR_CICLO / "contexto.json"


def preparar_ambiente():
    """Usado na 1ª célula dos notebooks: garante execução a partir da raiz do repositório."""
    os.chdir(RAIZ)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    return RAIZ


def carregar_config():
    with open(RAIZ / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def novo_cycle_id():
    agora = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = os.environ.get("GITHUB_RUN_NUMBER")
    return f"c{agora}" + (f"-gh{run}" if run else "-local")


def salvar_contexto(dados: dict):
    DIR_CICLO.mkdir(parents=True, exist_ok=True)
    atual = carregar_contexto() if ARQ_CONTEXTO.exists() else {}
    atual.update(dados)
    ARQ_CONTEXTO.write_text(json.dumps(atual, indent=2, ensure_ascii=False), encoding="utf-8")
    return atual


def carregar_contexto():
    if not ARQ_CONTEXTO.exists():
        raise RuntimeError("Contexto do ciclo não encontrado. Execute o notebook 01 primeiro.")
    return json.loads(ARQ_CONTEXTO.read_text(encoding="utf-8"))


def dir_evidencias(cycle_id: str) -> Path:
    d = RAIZ / "evidencias" / cycle_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def gatilho_execucao():
    return os.environ.get("GITHUB_EVENT_NAME", "local")


# ---------------------------------------------------------------- MLflow
def _corrigir_caminhos_mlflow(db: Path, novo: str) -> None:
    """O MLflow grava caminhos absolutos de artefatos. Como o runner do GitHub e a máquina
    local usam pastas diferentes, reescrevemos o prefixo para a raiz atual do repositório."""
    if not db.exists():
        return
    con = sqlite3.connect(db)
    try:
        cur = con.cursor()
        tabelas = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if "experiments" not in tabelas:
            return
        locs = [r[0] for r in cur.execute("SELECT artifact_location FROM experiments") if r[0]]
        antigos = set()
        for loc in locs:
            if "mlartifacts" in loc:
                antigos.add(loc[: loc.index("mlartifacts") + len("mlartifacts")])
        antigos.discard(novo)
        for antigo in antigos:
            for tab in tabelas:
                tipos_texto = ("TEXT", "VARCHAR", "CHAR")
                cols = [
                    c[1] for c in cur.execute(f'PRAGMA table_info("{tab}")')
                    if any(t in (c[2] or "").upper() for t in tipos_texto)
                ]
                for col in cols:
                    cur.execute(
                        f'UPDATE "{tab}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE "{col}" LIKE ?',
                        (antigo, novo, f"%{antigo}%"),
                    )
        con.commit()
    finally:
        con.close()


def configurar_mlflow(cfg=None):
    import mlflow

    cfg = cfg or carregar_config()
    db = RAIZ / cfg["mlflow"]["tracking_uri"].replace("sqlite:///", "")
    artefatos = RAIZ / cfg["mlflow"]["artifacts_dir"]
    artefatos.mkdir(exist_ok=True)
    _corrigir_caminhos_mlflow(db, artefatos.as_uri())
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    nome = cfg["mlflow"]["experimento"]
    if mlflow.get_experiment_by_name(nome) is None:
        mlflow.create_experiment(nome, artifact_location=artefatos.as_uri())
    mlflow.set_experiment(nome)
    return mlflow


def preparar_mlflow_para_publicacao() -> None:
    """Retira o caminho da máquina do arquivo que será versionado."""
    cfg = carregar_config()["mlflow"]
    db = RAIZ / cfg["tracking_uri"].removeprefix("sqlite:///")
    _corrigir_caminhos_mlflow(db, f"file:///{cfg['artifacts_dir']}")
