## Ciclo `c20261124T230040Z-local`

| Item | Valor |
|---|---|
| Gatilho | `local` |
| Lote incorporado | lote_04.csv |
| data_version | `v4-16552413` |
| Linhas de treino | 670 |
| Features eleitas | relacao_agua_aglomerante, log_idade, cimento, idade, aglomerante_total, relacao_agua_cimento |
| Modelo A (ridge) – MAE val | 6.157 |
| Modelo B (random_forest) – MAE val | 4.816 |
| Vencedor do ciclo | random_forest |
| MAE do vigente (antes) | 5.7281 |
| **Decisão** | **PROMOVIDO** – melhoria de 15.9% >= 2% (MAE 5.728 -> 4.816) |
| Drift | 7/9 variáveis com drift (KS, p<0.05) – ALERTA |

> ⚠️ **Alerta de drift**: 7/9 variáveis com drift (KS, p<0.05) – ALERTA. Revise o lote antes de confiar nas previsões.
