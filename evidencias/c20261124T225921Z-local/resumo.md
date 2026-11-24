## Ciclo `c20261124T225921Z-local`

| Item | Valor |
|---|---|
| Gatilho | `local` |
| Lote incorporado | lote_02.csv |
| data_version | `v2-075a21dd` |
| Linhas de treino | 442 |
| Features eleitas | aglomerante_total, log_idade, relacao_agua_cimento, relacao_agua_aglomerante, idade |
| Modelo A (ridge) – MAE val | 6.046 |
| Modelo B (random_forest) – MAE val | 5.728 |
| Vencedor do ciclo | random_forest |
| MAE do vigente (antes) | 6.209 |
| **Decisão** | **PROMOVIDO** – melhoria de 7.7% >= 2% (MAE 6.209 -> 5.728) |
| Drift | 2/9 variáveis com drift (KS, p<0.05) |
