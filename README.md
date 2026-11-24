# Previsão da resistência à compressão do concreto

Projeto N3 · Análise Preditiva · Católica de Santa Catarina

Professor Rodrigo Hagstrom

## Problema e dados

Estimamos a **resistência à compressão do concreto, em MPa**, a partir da composição da mistura e da idade. Concreteiras, laboratórios e equipes de controle de qualidade podem usar a estimativa para comparar traços e priorizar ensaios. A confirmação da resistência exige ensaio de laboratório.

A base [Concrete Compressive Strength](https://doi.org/10.24432/C5PK67), de I-Cheng Yeh, vem da UCI e tem licença CC BY 4.0. Ela reúne 1.030 ensaios com as quantidades dos sete ingredientes (kg/m³), a idade (dias) e a resistência medida (MPa). A resistência varia de **2,33 a 82,60 MPa**. As idades vão de **1 a 365 dias**, com 425 ensaios aos 28 dias. A análise dos lotes registra valores ausentes, duplicatas, violações de faixa e valores extremos.

## Demonstração

Use Python 3.13 ou 3.14. Na raiz do repositório, crie e ative um ambiente virtual e instale as dependências.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.prever
python -m src.prever --idade 90
```

No PowerShell, ative o ambiente com `.venv\Scripts\Activate.ps1`. Com o modelo incluído na entrega, o exemplo retorna **41,96 MPa aos 28 dias** e **55,03 MPa aos 90 dias** para a mesma composição. O script valida os valores de entrada. Use `python -m src.prever --help` para ver os parâmetros e seus valores padrão.

## Como treinamos e avaliamos

1. **Preparação.** Removemos 25 duplicatas exatas, mantivemos os extremos plausíveis e usamos quatro lotes para simular a chegada de dados. O ETL rejeita valores fora das faixas e remove duplicatas. Valores ausentes em variáveis opcionais são imputados pela mediana, calculada no treino.
2. **Divisão por traço.** As diferentes idades de uma composição ficam na mesma partição. A chave usa os seis ingredientes sempre observados, sem o superplastificante, que pode estar ausente. Essa escolha agrupa até traços que diferem só no aditivo. Ficaram **670 ensaios no treino, 131 na validação e 204 no teste**, sem grupos em comum.
3. **Modelos.** Comparamos Ridge e Random Forest com um baseline que prevê a média. Ambos usam os mesmos dados e atributos eleitos por três métodos de seleção. Entre os atributos derivados estão a relação água/cimento, a relação água/aglomerante, o total de aglomerante e a transformação logarítmica da idade. A validação cruzada também separa por traço e refaz a seleção de atributos em cada fold.
4. **Escolha e teste.** O menor MAE na validação fixa define o vencedor. Para substituir um modelo vigente, o desafiante precisa superar o baseline e melhorar o MAE em pelo menos 2%. O teste reservado é usado para medir o desempenho, fora das decisões de treino e promoção.

## Resultados

| No teste reservado (204 ensaios de 85 grupos inéditos) | Resultado |
|---|---:|
| MAE do Random Forest | **4,19 MPa** |
| MAE da previsão pela média (baseline) | 13,64 MPa |
| RMSE do Random Forest | 6,01 MPa |
| R² do Random Forest | 0,869 |

O Random Forest foi promovido no quarto ciclo, com MAE de 4,82 MPa na validação. No quinto ciclo, sem dados novos, continuou vigente. Em concretos com resistência medida acima de 60 MPa, o erro absoluto médio foi de **6,72 MPa** e a previsão ficou, em média, **4,59 MPa abaixo** da medição (17 ensaios).

O **MAE** é o erro absoluto médio, em MPa. O **RMSE** dá mais peso aos erros maiores. As estimativas foram avaliadas com uma base de laboratório. Materiais locais, tipo de cimento, cura e temperatura podem alterar o desempenho.

O [relatório](reports/relatorio_n3.pdf) detalha os experimentos, os erros e as referências. O [histórico](evidencias/historico_ciclos.csv) registra as decisões de cada ciclo. O [manifesto](models/modelo_vigente.json) identifica o modelo usado na demonstração.

## Reproduzir o pipeline

```bash
bash scripts/executar_ciclo.sh
```

Cada execução roda os cinco notebooks em ordem e incorpora um lote, quando disponível. O checkout já traz os quatro lotes processados e o modelo vigente. Para repetir desde o primeiro lote, use **uma cópia de trabalho separada** e execute `python scripts/reset_estado.py`, que apaga o histórico e o estado dessa cópia. Em seguida, rode o comando de ciclo quatro vezes e, para demonstrar a manutenção, mais uma vez sem lote novo. Para regenerar os lotes a partir da base da UCI, execute `python -m scripts.preparar_lotes` antes do primeiro ciclo.

O [workflow](.github/workflows/pipeline_mlops.yml) permite execução manual e agendamento semanal. A revisão por traço foi reexecutada localmente. Para consultar o MLflow, rode `python -c "from src.utils import configurar_mlflow; configurar_mlflow()"` e depois `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.

## Roteiro para a apresentação — 1º de dezembro

1. **Problema.** “Recebemos a composição e a idade do concreto. O modelo estima a resistência em MPa.”
2. **Experimento.** “Agrupamos ensaios da mesma composição para testar em traços inéditos. Comparamos dois modelos com a previsão pela média.”
3. **Resultado.** Mostrem o MAE e o gráfico previsto × real do relatório. Expliquem onde o erro aumenta e que o uso é de triagem.
4. **Demonstração.** Executem os exemplos de 28 e 90 dias e expliquem como interpretar a estimativa. Cada integrante apresenta a parte em que trabalhou.

## Equipe

Luciano L. Zimmermann · Luiz Eduardo Uber · Matheus Lorenzetti · Rafael Gadotti · Yuri J. Pereira

## Referências

- YEH, I.-C. *Modeling of strength of high-performance concrete using artificial neural networks*. Cement and Concrete Research, 28(12), 1797–1808, 1998.
- YEH, I.-C. *Concrete Compressive Strength*. UCI Machine Learning Repository, 1998. [Dataset e licença](https://doi.org/10.24432/C5PK67).
- PEDREGOSA, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830, 2011.
