## Ciclo `c20261124T230002Z-local`

| Item | Valor |
|---|---|
| Gatilho | `local` |
| Lote incorporado | lote_03.csv |
| data_version | `v3-c6be3d56` |
| Linhas de treino | 542 |
| Features eleitas | relacao_agua_cimento, log_idade, relacao_agua_aglomerante, cimento, aglomerante_total, idade |
| Modelo A (ridge) – MAE val | 6.157 |
| Modelo B (random_forest) – MAE val | 5.784 |
| Vencedor do ciclo | random_forest |
| MAE do vigente (antes) | 5.7281 |
| **Decisão** | **MANTIDO** – melhoria de -1.0% < 2% exigidos (vigente 5.728 x desafiante 5.784) |
| Drift | 1/9 variáveis com drift (KS, p<0.05) |
