"""Fecha o ciclo: guarda notebooks executados como evidência, gera resumo e alerta de drift.

Uso (ao fim do workflow): python -m src.finalizar_ciclo
"""
import json
import os
import shutil

from .utils import RAIZ, carregar_contexto, dir_evidencias, preparar_mlflow_para_publicacao


def main():
    ctx = carregar_contexto()
    cid = ctx["cycle_id"]
    ev = dir_evidencias(cid)

    origem = RAIZ / "artifacts" / "notebooks_executados"
    if origem.exists():
        destino = ev / "notebooks"
        destino.mkdir(exist_ok=True)
        for nb in origem.glob("*.ipynb"):
            shutil.copy2(nb, destino / nb.name)

    dec = json.loads((ev / "decisao.json").read_text(encoding="utf-8"))
    linhas = [
        f"## Ciclo `{cid}`",
        "",
        "| Item | Valor |", "|---|---|",
        f"| Gatilho | `{ctx.get('gatilho')}` |",
        f"| Lote incorporado | {ctx.get('lote') or 'nenhum (sem dados novos)'} |",
        f"| data_version | `{ctx['data_version']}` |",
        f"| Linhas de treino | {ctx['n_treino']} |",
        f"| Features eleitas | {', '.join(ctx['features_eleitas'])} |",
        f"| Modelo A (ridge) – MAE val | {ctx['resultados']['A']['val_mae']:.3f} |",
        f"| Modelo B (random_forest) – MAE val | {ctx['resultados']['B']['val_mae']:.3f} |",
        f"| Vencedor do ciclo | {dec['vencedor_algoritmo']} |",
        f"| MAE do vigente (antes) | {dec['mae_vigente_antes'] if dec['mae_vigente_antes'] is not None else '—'} |",
        f"| **Decisão** | **{dec['decisao'].upper()}** – {dec['motivo']} |",
        f"| Drift | {ctx.get('drift_resumo') or 'não avaliado (primeiro ciclo ou sem lote novo)'} |",
    ]
    if ctx.get("drift_alerta"):
        linhas += ["", f"> ⚠️ **Alerta de drift**: {ctx['drift_resumo']}. Revise o lote antes de confiar nas previsões."]
        print(f"::warning title=Drift detectado::{ctx['drift_resumo']}")
    texto = "\n".join(linhas) + "\n"
    (ev / "resumo.md").write_text(texto, encoding="utf-8")
    print(texto)

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(texto)
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as f:
            f.write(f"CYCLE_ID={cid}\nDECISAO={dec['decisao']}\n")
    for resultado in ctx["resultados"].values():
        resultado.pop("arquivo", None)
    (ev / "contexto.json").write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    preparar_mlflow_para_publicacao()


if __name__ == "__main__":
    main()
