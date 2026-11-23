## Ciclo `c20261123T090417Z-gh4`

| Item | Valor |
|---|---|
| Gatilho | `workflow_dispatch` |
| Lote incorporado | lote_04.csv |
| data_version | `v4-6e425ef4` |
| Linhas de treino | 653 |
| Features eleitas | aglomerante_total, log_idade, relacao_agua_aglomerante, idade, relacao_agua_cimento |
| Modelo A (ridge) – MAE val | 6.120 |
| Modelo B (random_forest) – MAE val | 4.264 |
| Vencedor do ciclo | random_forest |
| MAE do vigente (antes) | 4.56 |
| **Decisão** | **PROMOVIDO** – melhoria de 6.5% >= 2% (MAE 4.560 -> 4.264) |
| Drift | 7/9 variáveis com drift (KS, p<0.05) – ALERTA |

> ⚠️ **Alerta de drift**: 7/9 variáveis com drift (KS, p<0.05) – ALERTA. Revise o lote antes de confiar nas previsões.
