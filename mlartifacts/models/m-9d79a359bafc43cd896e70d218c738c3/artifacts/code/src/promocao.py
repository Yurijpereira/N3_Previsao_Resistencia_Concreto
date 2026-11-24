"""Eleição do vencedor do ciclo e regra de promoção do modelo vigente."""
import json
import shutil
from datetime import UTC, datetime

import joblib
import pandas as pd

from .modelagem import metricas
from .utils import RAIZ

MANIFESTO = RAIZ / "models" / "modelo_vigente.json"
HISTORICO_MODELOS = RAIZ / "models" / "historico"
HISTORICO_CICLOS = RAIZ / "evidencias" / "historico_ciclos.csv"


def eleger_vencedor(resultados: dict, metrica="val_mae") -> str:
    """Menor MAE na validação fixa. Empate -> modelo A (mais simples, Ridge)."""
    return min(sorted(resultados), key=lambda k: resultados[k][metrica])


def carregar_manifesto() -> dict | None:
    return json.loads(MANIFESTO.read_text(encoding="utf-8")) if MANIFESTO.exists() else None


def avaliar_vigente(validacao: pd.DataFrame, cfg):
    """Reavalia o modelo vigente (com seu próprio pré-processamento e features) na MESMA
    validação fixa usada pelos candidatos -> protocolo compatível entre ciclos."""
    manifesto = carregar_manifesto()
    if manifesto is None:
        return None, None
    modelo = joblib.load(RAIZ / manifesto["arquivo_modelo"])
    alvo = cfg["dados"]["alvo"]
    m = metricas(validacao[alvo], modelo.predict(validacao.drop(columns=[alvo])))
    return manifesto, m


def decidir(mae_desafiante, mae_vigente, mae_baseline, cfg):
    p = cfg["promocao"]
    if p["exigir_melhor_que_baseline"] and mae_desafiante >= mae_baseline:
        return False, f"desafiante (MAE {mae_desafiante:.3f}) não supera o baseline ({mae_baseline:.3f})"
    if mae_vigente is None:
        return True, "não há modelo vigente: primeiro modelo é promovido"
    if mae_vigente == 0:
        return False, "o modelo vigente já tem erro zero na validação"
    limite = mae_vigente * (1 - p["melhoria_minima_relativa"])
    melhoria = (mae_vigente - mae_desafiante) / mae_vigente
    if mae_desafiante <= limite:
        return True, f"melhoria de {melhoria:.1%} >= {p['melhoria_minima_relativa']:.0%} (MAE {mae_vigente:.3f} -> {mae_desafiante:.3f})"
    return False, f"melhoria de {melhoria:.1%} < {p['melhoria_minima_relativa']:.0%} exigidos (vigente {mae_vigente:.3f} x desafiante {mae_desafiante:.3f})"


def promover(arquivo_candidato, info: dict) -> dict:
    HISTORICO_MODELOS.mkdir(parents=True, exist_ok=True)
    destino = HISTORICO_MODELOS / f"modelo_{info['cycle_id']}_{info['algoritmo']}.joblib"
    shutil.copy2(arquivo_candidato, destino)  # modelos anteriores são preservados
    anterior = carregar_manifesto()
    manifesto = {
        **info,
        "arquivo_modelo": str(destino.relative_to(RAIZ)),
        "promovido_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "anterior": None if anterior is None else {
            k: anterior.get(k) for k in ("cycle_id", "algoritmo", "data_version", "val_mae", "run_id", "arquivo_modelo")
        },
    }
    return manifesto


def salvar_manifesto(manifesto: dict):
    MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    MANIFESTO.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")


def registrar_historico(linha: dict):
    HISTORICO_CICLOS.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([linha])
    if HISTORICO_CICLOS.exists():
        df = pd.concat([pd.read_csv(HISTORICO_CICLOS), df], ignore_index=True)
    df.to_csv(HISTORICO_CICLOS, index=False)
