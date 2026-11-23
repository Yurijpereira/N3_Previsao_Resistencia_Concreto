# Trabalho N3 · Análise Preditiva · Pipeline MLOps

**Previsão da resistência à compressão do concreto** com um pipeline único e agendado que, a cada ciclo:
incorpora um novo lote de dados → faz ETL e validação → monitora drift → elege features por votação de 3 métodos →
treina 2 algoritmos (Ridge × Random Forest) → elege o vencedor → decide **promover ou manter** o modelo vigente →
registra tudo no MLflow → gera previsões e evidências.

Católica de Santa Catarina · Prof. M. Sc. Rodrigo Hagstrom
**Integrantes:** Luciano L. Zimmermann · Luiz Eduardo Uber · Matheus Lorenzetti · Rafael Gadotti · Yuri J. Pereira

---

## 1. Problema e dados
| | |
|---|---|
| Tarefa | Regressão: `resistencia` (MPa) |
| Dados | [UCI – Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (Yeh, 1998), licença CC BY 4.0. 1.030 linhas, sendo 1.005 após remover duplicatas |
| Atributos | cimento, escória, cinza volante, água, superplastificante, agregado graúdo, agregado miúdo (kg/m³) e idade (dias), mais 4 derivados |
| Métrica principal | **MAE** (MPa), a minimizar |
| Particionamento | teste reservado 20% · validação fixa 15% · 4 lotes de treino que chegam um por ciclo |

`scripts/preparar_lotes.py` gera, uma única vez, `data/holdout/` e `data/raw/lote_01..04.csv`. Os lotes trazem problemas
propositais para o ETL tratar: ausentes no lote 2, duplicatas e uma linha inválida no lote 3, e uma mudança de perfil (drift) no lote 4.

## 2. Estrutura
```
.github/workflows/pipeline_mlops.yml   # workflow ÚNICO (schedule + workflow_dispatch)
config/config.yaml                     # dados, métodos, votação, algoritmos, métrica, limiar de promoção
data/raw/                              # lotes que chegam a cada ciclo
data/holdout/                          # validacao.csv (decisões) e teste.csv (reservado)
data/processed/treino_acumulado.csv    # treino incorporado até o ciclo atual
data/state/estado_ingestao.json        # lotes já processados + ciclos
notebooks/01_etl.ipynb … 05_previsoes.ipynb
src/                                   # ingestao, preprocessamento, features, modelagem, promocao, drift
models/modelo_vigente.json             # manifesto do modelo vigente
models/historico/                      # todos os modelos já promovidos (preservados)
mlflow.db + mlartifacts/               # tracking MLflow (versionado no repositório)
evidencias/<cycle_id>/                 # votação, comparação, decisão, drift, previsões, notebooks executados
evidencias/historico_ciclos.csv        # uma linha por ciclo
reports/relatorio_n3.pdf
scripts/                               # preparar_lotes, executar_ciclo (local), reset_estado, gerar_relatorio
```

## 3. Instalação e execução local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/executar_ciclo.sh        # roda 01→05 + evidências (1 ciclo = 1 lote novo)
```
Cada execução incorpora o próximo lote. Sem lote novo, o ciclo roda com os mesmos dados (mesmo `data_version`) e o
vigente tende a ser mantido. Para recomeçar do zero: `python scripts/reset_estado.py`.

No Windows sem bash, rode os comandos `jupyter nbconvert --to notebook --execute notebooks/0X_....ipynb` em ordem e depois `python -m src.finalizar_ciclo`.

## 4. Execução agendada (GitHub Actions)
1. Suba o repositório no GitHub (branch padrão `main`).
2. Vá em **Settings → Actions → General → Workflow permissions** e marque **Read and write permissions**.
3. O workflow roda pelo `cron`. Durante a coleta de evidências ele rodou a cada 30 min (`17,47 * * * *`). Agora está semanal (`23 9 * * 1`, segunda 06:23 BRT). O GitHub pode atrasar o agendamento.
4. Ao fim de cada execução, o próprio workflow faz **commit do estado** (MLflow, modelo vigente, evidências) e publica
   um **artifact** `resultados-ciclo-N`. O próximo ciclo parte desse estado; nada depende do runner anterior.
5. O **resumo da execução** (aba *Summary*) mostra a decisão do ciclo e o **alerta de drift**, quando houver.
6. Para demonstrar de novo, use **Run workflow** (workflow_dispatch) ou volte o cron para uma frequência maior.

Se qualquer etapa falhar, o job para antes do commit, e **nenhuma promoção é persistida**.

### Resultados obtidos no GitHub Actions
| Ciclo | Gatilho | Lote | MAE val Ridge | MAE val RF | Decisão | Drift |
|---|---|---|---|---|---|---|
| 1 | workflow_dispatch | lote_01 | 7,794 | 5,331 | promovido (1º modelo) | - |
| 2 | **schedule** | lote_02 | 7,904 | 4,938 | promovido (+7,4%) | não |
| 3 | **schedule** | lote_03 | 7,722 | 4,560 | promovido (+7,7%) | não |
| 4 | workflow_dispatch | lote_04 | 6,120 | 4,264 | promovido (+6,5%) | **alerta (7/9 variáveis)** |
| 5 | **schedule** | sem lote novo | 6,120 | 4,264 | **mantido** (0% < 2%) | - |

O modelo vigente final é o Random Forest do ciclo 4, com MAE de **4,67 MPa** e R² de 0,867 no teste reservado. Os detalhes de cada ciclo estão em `evidencias/<cycle_id>/` e `evidencias/historico_ciclos.csv`. O relatório completo está em `reports/relatorio_n3.pdf`, gerado por `python scripts/gerar_relatorio.py --repo <link>`.

## 5. Consultar o MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # http://localhost:5000
```
Experimento `n3-resistencia-concreto`. Todos os runs têm as tags `cycle_id`, `data_version` e `etapa`
(`etl`, `votacao`, `treino`, `decisao`, `previsoes`). Filtro útil: `tags.cycle_id = '<id>'`.
Antes de abrir a UI, rode qualquer notebook (ou `python -c "from src.utils import configurar_mlflow; configurar_mlflow()"`):
os caminhos dos artefatos são ajustados para a pasta local automaticamente.

## 6. Reproduzir previsões
```python
import joblib, json, pandas as pd
m = json.load(open("models/modelo_vigente.json"))
modelo = joblib.load(m["arquivo_modelo"])          # Pipeline com pré-processamento + features eleitas
nova = pd.DataFrame([{"cimento": 350, "escoria": 100, "cinza_volante": 0, "agua": 180,
                      "superplastificante": 6, "agregado_graudo": 970, "agregado_miudo": 780, "idade": 28}])
print(modelo.predict(nova))
```
(execute a partir da raiz do repositório, pois o pipeline usa `src.preprocessamento.criar_atributos`).

## 7. Regras definidas antes do treinamento
- **Votação:** cada método (SelectKBest f_regression, importância RF, RFE-Ridge) vota no seu top-5. A feature é eleita com ≥ 2 votos. O mínimo é 4 features, completando pelo menor ranking médio.
- **Competição:** mesmas features, mesmo treino e mesma seed. Vence o menor MAE na validação fixa (empate → Ridge).
- **Promoção:** o desafiante precisa superar o baseline (média) **e** ser ≥ 2% melhor que o vigente reavaliado na mesma validação.
- **Drift (recurso extra):** teste KS por variável (α = 0,05). Gera alerta se ≥ 25% das variáveis apresentarem drift.
