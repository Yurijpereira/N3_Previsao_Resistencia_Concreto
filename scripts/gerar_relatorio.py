"""Gera reports/relatorio_n3.pdf a partir das evidências reais dos ciclos executados.

Uso:
    python scripts/gerar_relatorio.py --repo https://github.com/usuario/projeto-n3

Capturas de tela (ex.: aba Actions mostrando execuções 'schedule', MLflow UI) colocadas em
reports/prints/*.png são incluídas automaticamente na seção de evidências, em ordem alfabética.
O nome do arquivo vira a legenda (ex.: 01_actions_schedule.png -> "actions schedule").
"""
import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
EV = RAIZ / "evidencias"
INTEGRANTES = ["Luciano L. Zimmermann", "Luiz Eduardo Uber", "Matheus Lorenzetti", "Rafael Gadotti", "Yuri J. Pereira"]
AZUL = colors.HexColor("#1f4e79")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=AZUL, spaceBefore=10, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, textColor=AZUL, spaceBefore=8, spaceAfter=4, keepWithNext=1)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=10.5, leading=14.5, spaceAfter=5)
PL = ParagraphStyle("PL", parent=P, leftIndent=12, bulletIndent=2)
CAP = ParagraphStyle("CAP", parent=P, fontSize=9, textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
CEL = ParagraphStyle("CEL", parent=P, fontSize=8.5, leading=10.5, spaceAfter=0)


def tabela(dados, larguras=None, cabecalho=True):
    linhas = [[Paragraph(str(c), CEL) for c in linha] for linha in dados]
    t = Table(linhas, colWidths=larguras, repeatRows=1 if cabecalho else 0)
    estilo = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b7c3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if cabecalho:
        estilo += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1"))]
    t.setStyle(TableStyle(estilo))
    return t


def figura(caminho, largura_cm, legenda):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(caminho)).getSize()
    largura = largura_cm * cm
    return KeepTogether([Image(str(caminho), width=largura, height=largura * h / w), Paragraph(legenda, CAP), Spacer(1, 6)])


def bullets(itens):
    return [Paragraph(i, PL, bulletText="•") for i in itens]


def f3(x):
    return "-" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{float(x):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="", help="link do repositório GitHub")
    args = ap.parse_args()

    hist = pd.read_csv(EV / "historico_ciclos.csv")
    ciclos = [EV / c for c in hist["cycle_id"]]
    ctxs = [json.loads((c / "contexto.json").read_text(encoding="utf-8")) for c in ciclos]
    decs = [json.loads((c / "decisao.json").read_text(encoding="utf-8")) for c in ciclos]
    com_lote = [i for i, c in enumerate(ctxs) if c.get("lote")]
    i_ult = com_lote[-1]                           # último ciclo com dados novos
    c_ult = ciclos[i_ult]
    i_drift = next((i for i, c in enumerate(ctxs) if c.get("drift_alerta")), None)
    manifesto = json.loads((RAIZ / "models/modelo_vigente.json").read_text(encoding="utf-8"))
    gatilhos = hist["gatilho"].value_counts().to_dict()

    doc = SimpleDocTemplate(str(RAIZ / "reports/relatorio_n3.pdf"), pagesize=A4, leftMargin=2.2 * cm,
                            rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Relatório Técnico - Trabalho N3 - Análise Preditiva", author=", ".join(INTEGRANTES))
    s = []

    # ------------------------------------------------------------ capa
    s += [Spacer(1, 3 * cm),
          Paragraph("Católica de Santa Catarina", ParagraphStyle("c0", parent=P, alignment=TA_CENTER, fontSize=12)),
          Paragraph("Análise Preditiva · Trabalho Avaliativo N3", ParagraphStyle("c1", parent=P, alignment=TA_CENTER, fontSize=12)),
          Spacer(1, 1.2 * cm),
          Paragraph("Pipeline MLOps para previsão da resistência à compressão do concreto",
                    ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=25, textColor=AZUL)),
          Paragraph("Votação de features, competição de modelos e promoção automatizada",
                    ParagraphStyle("st", parent=P, alignment=TA_CENTER, fontSize=12.5)),
          Spacer(1, 1.5 * cm),
          Paragraph("<b>Integrantes</b><br/>" + "<br/>".join(INTEGRANTES), ParagraphStyle("i", parent=P, alignment=TA_CENTER, fontSize=11.5, leading=17)),
          Spacer(1, 1 * cm),
          Paragraph("Prof. M. Sc. Rodrigo Hagstrom", ParagraphStyle("p", parent=P, alignment=TA_CENTER)),
          Paragraph(f"Repositório: {args.repo or '(link do repositório)'}", ParagraphStyle("r", parent=P, alignment=TA_CENTER)),
          Paragraph(f"Jaraguá do Sul, {date.today():%d/%m/%Y}", ParagraphStyle("d", parent=P, alignment=TA_CENTER)),
          PageBreak()]

    # ------------------------------------------------------------ 1 problema
    s += [Paragraph("1. Problema, dados e desenho da avaliação", H1),
          Paragraph("O objetivo é prever a <b>resistência à compressão do concreto (MPa)</b> a partir da composição da mistura e da "
                    "idade do corpo de prova. Trata-se de um problema de <b>regressão</b> com aplicação direta no controle de qualidade: "
                    "estimar a resistência antes do ensaio destrutivo ajuda a ajustar traços e reduzir desperdício.", P),
          tabela([
              ["Item", "Definição"],
              ["Fonte e licença", "UCI Machine Learning Repository - Concrete Compressive Strength (I-Cheng Yeh, 1998). Licença CC BY 4.0."],
              ["Unidade de observação", "Um corpo de prova: mistura (kg/m³ de cada componente) + idade de ensaio (dias)."],
              ["Variável-alvo", "resistencia (MPa)."],
              ["Atributos", "cimento, escoria, cinza_volante, agua, superplastificante, agregado_graudo, agregado_miudo, idade + 4 derivados."],
              ["Tamanho", "1.030 registros, 25 duplicatas exatas removidas antes da divisão -> 1.005 registros únicos."],
              ["Métrica principal", "MAE (erro absoluto médio, MPa), a minimizar. Está na unidade do problema, é fácil de comunicar e é menos sensível a poucos erros grandes que o RMSE. RMSE e R<super>2</super> são registrados como apoio."],
              ["Particionamento", "Com seed fixa: 20% teste reservado (201) · 15% validação fixa (151) · 653 restantes divididos em 4 lotes que chegam um por ciclo."],
              ["Chegada de novos dados", "Cada execução agendada incorpora o próximo lote de data/raw/ ainda não processado (controle em data/state/estado_ingestao.json)."],
          ], [4.2 * cm, 12.4 * cm]),
          Spacer(1, 6),
          Paragraph("<b>Protocolo.</b> O conjunto de <b>teste</b> nunca participa de decisões. Ele só é usado no notebook 05 para "
                    "medir o desempenho final do modelo vigente. Todas as decisões recorrentes (vencedor do ciclo e promoção) usam a "
                    "<b>validação fixa</b>, que é igual em todos os ciclos e por isso torna o vigente e o desafiante comparáveis. A "
                    "validação cruzada de 5 folds no treino é registrada como informação complementar.", P),
          Paragraph("<b>Simulação dos lotes.</b> O dataset não tem ordem temporal, então a chegada de dados foi simulada. Os lotes 1 a 3 são "
                    "aleatórios. O lote 4 reúne os corpos de prova com idade >= 90 dias e simula uma mudança de perfil dos dados. Foram "
                    "injetados problemas controlados para exercitar o ETL: 3 valores ausentes no lote 2, além de 5 linhas repetidas do lote 1 "
                    "e 1 linha com água negativa no lote 3.", P)]

    # ------------------------------------------------------------ 2 ETL
    linhas_etl = [["Ciclo", "Lote", "Recebidas", "Fora da faixa", "Já incorporadas", "Ausentes (imputar)", "Aceitas", "Treino acumulado"]]
    for k, c in enumerate(ctxs, 1):
        e = c["etl"]
        linhas_etl.append([k, c.get("lote") or "nenhum", e.get("linhas_recebidas", 0), e.get("removidas_fora_da_faixa", 0),
                           e.get("removidas_ja_incorporadas", 0), e.get("ausentes_para_imputacao", 0), e.get("linhas_aceitas", 0), c["n_treino"]])
    s += [Paragraph("2. ETL e engenharia de atributos", H1),
          Paragraph("O notebook 01 carrega o lote do ciclo e faz um diagnóstico de ausentes, duplicatas, violações de faixa e outliers (regra "
                    "IQR). Em seguida, aplica as regras definidas em <i>config.yaml</i>:", P)]
    s += bullets([
        "<b>Esquema:</b> se faltar alguma coluna, o ciclo falha e nada é promovido.",
        "<b>Faixas físicas plausíveis</b> por variável. Linhas inválidas são rejeitadas, como a de água negativa no lote 3.",
        "<b>Duplicatas:</b> são removidas dentro do lote e também em relação ao que já foi incorporado (hash por linha), para que o mesmo dado não entre duas vezes.",
        "<b>Ausentes:</b> são mantidos e imputados pela mediana <i>dentro do Pipeline</i>, ajustado somente no treino, o que evita vazamento.",
        "<b>Outliers (IQR):</b> são apenas reportados. Idades de 180 a 365 dias e resistências altas são reais e importantes para o modelo.",
    ])
    s += [tabela(linhas_etl, [1.2 * cm, 2.1 * cm, 1.9 * cm, 2 * cm, 2.2 * cm, 2.3 * cm, 1.6 * cm, 2.4 * cm]),
          Paragraph("Tabela 1 - Resultado do ETL em cada ciclo (fonte: evidencias/&lt;cycle_id&gt;/contexto.json).", CAP),
          Paragraph("<b>Atributos derivados</b> (determinísticos, sem parâmetros ajustados): <i>relacao_agua_cimento</i> (lei de Abrams), "
                    "<i>aglomerante_total</i> (cimento + escória + cinza), <i>relacao_agua_aglomerante</i> e <i>log_idade</i> (o ganho de "
                    "resistência é aproximadamente logarítmico no tempo). Todo o pré-processamento fica em um <i>Pipeline</i> do "
                    "scikit-learn (FunctionTransformer -> ColumnTransformer com as features eleitas -> SimpleImputer -> StandardScaler) "
                    "serializado junto com o modelo. Assim, a previsão aplica exatamente as mesmas transformações.", P),
          Paragraph("<b>Versionamento.</b> Cada ciclo recebe um <i>cycle_id</i> (data/hora UTC + número da execução no GitHub) e um "
                    "<i>data_version</i> (quantidade de lotes + hash SHA-256 do treino acumulado). Esses identificadores aparecem em todos os "
                    "runs do MLflow, nas evidências e no manifesto.", P)]

    # ------------------------------------------------------------ 3 votação
    vot = pd.read_csv(c_ult / "votacao_features.csv")
    lv = [["Feature", "SelectKBest", "RF", "RFE", "Votos", "Rank médio", "Eleita"]]
    for _, r in vot.iterrows():
        lv.append([r["feature"], r["voto_kbest"], r["voto_rf"], r["voto_rfe"], int(r["votos"]), f"{r['ranking_medio']:.2f}", "sim" if r["eleita"] else "-"])
    s += [Paragraph("3. Três métodos de seleção e votação de features", H1),
          Paragraph("O notebook 02 executa três abordagens sobre o treino acumulado, sem usar a validação nem o teste:", P)]
    s += bullets([
        "<b>SelectKBest</b> com <i>f_regression</i> (estatística F univariada);",
        "<b>Importância por Random Forest</b> (300 árvores, redução média de impureza);",
        "<b>RFE</b> com <i>Ridge</i> sobre atributos padronizados.",
    ])
    s += [Paragraph("<b>Regra de votação (fixada antes da comparação):</b> cada método vota em suas 5 melhores features. É eleita a feature "
                    "aprovada por pelo menos 2 dos 3 métodos. Se forem eleitas menos de 4, completa-se pelo menor ranking médio. Os votos de "
                    "cada feature, as listas de cada método, os parâmetros e o conjunto final são registrados no MLflow (run <i>votacao_*</i>).", P),
          tabela(lv, [4.6 * cm, 2 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 2.2 * cm, 1.6 * cm]),
          Paragraph(f"Tabela 2 - Votação do ciclo {i_ult + 1} ({ctxs[i_ult]['lote']}, {ctxs[i_ult]['n_treino']} linhas de treino).", CAP),
          figura(c_ult / "votacao_features.png", 12, f"Figura 1 - Votos por feature no ciclo {i_ult + 1}."),
          Paragraph("Ao longo dos ciclos, o conjunto eleito foi: " + "; ".join(
              f"ciclo {k}: {f.replace('|', ', ')}" for k, f in enumerate(hist["features_eleitas"], 1)) + ". "
              "A votação favoreceu os atributos derivados, sobretudo <i>aglomerante_total</i>, <i>log_idade</i> e as relações "
              "água/cimento e água/aglomerante. Isso é coerente com a tecnologia do concreto e mostra que a engenharia de "
              "atributos agregou informação.", P)]

    # ------------------------------------------------------------ 4 competição
    comp = pd.read_csv(c_ult / "comparacao_modelos.csv", index_col=0)
    lc = [["Ciclo", "Lote", "n treino", "MAE Ridge (A)", "MAE RF (B)", "Vencedor"]]
    for k, r in enumerate(hist.itertuples(), 1):
        lc.append([k, r.lote, r.n_treino, f3(r.mae_ridge), f3(r.mae_random_forest), r.vencedor])
    s += [Paragraph("4. Competição entre dois algoritmos", H1),
          Paragraph("O notebook 03 treina o <b>Modelo A - Ridge</b> (alpha = 1,0) e o <b>Modelo B - Random Forest Regressor</b> "
                    "(100 árvores, max_depth = 12, min_samples_leaf = 2) com as <b>mesmas features eleitas</b>, o mesmo treino e a mesma "
                    "seed (42). Para cada modelo, ficam registrados no MLflow os hiperparâmetros, a seed, as features, as métricas de CV e de "
                    "validação e o modelo serializado. Os dois runs compartilham <i>cycle_id</i> e <i>data_version</i>. Um baseline que "
                    "prevê a média do treino (DummyRegressor) serve de piso.", P),
          tabela([["Modelo", "MAE val", "RMSE val", "R<super>2</super> val", "CV MAE (5 folds)"]] + [
              [f"{k} - {r['algoritmo']}", f3(r["val_mae"]), f3(r["val_rmse"]), f3(r["val_r2"]), f"{r['cv_mae_media']:.3f} ± {r['cv_mae_desvio']:.3f}"]
              for k, r in comp.iterrows()] + [["Baseline (média)", f3(decs[i_ult]["baseline_val_mae"]), "-", "-", "-"]],
              [4.4 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 3.6 * cm]),
          Paragraph(f"Tabela 3 - Comparação no ciclo {i_ult + 1} (validação fixa, {151} amostras).", CAP),
          tabela(lc, [1.3 * cm, 2.4 * cm, 1.9 * cm, 3 * cm, 3 * cm, 3 * cm]),
          Paragraph("Tabela 4 - MAE na validação fixa em todos os ciclos.", CAP),
          figura(c_ult / "comparacao_modelos.png", 10, f"Figura 2 - MAE dos candidatos e do baseline no ciclo {i_ult + 1}."),
          Paragraph("<b>Leitura dos resultados.</b> O Random Forest venceu em todos os ciclos. Um ponto interessante é que, nos primeiros "
                    "ciclos, o Ridge tinha CV no treino razoável, mas MAE alto na validação. Isso acontece porque o treino ainda não tinha "
                    "corpos de prova com mais de 56 dias e o modelo linear extrapola mal a idade, enquanto a validação fixa contém idades de "
                    "até 365 dias. Quando o lote 4 trouxe esses casos, o MAE do Ridge caiu bastante. Esse comportamento mostra por que a "
                    "decisão usa uma validação fixa e representativa, e não apenas a CV do treino.", P)]

    # ------------------------------------------------------------ 5 promoção
    lp = [["Ciclo", "data_version", "Vencedor (MAE)", "Vigente antes", "Decisão", "Vigente após"]]
    for k, (r, d) in enumerate(zip(hist.itertuples(), decs), 1):
        lp.append([k, r.data_version, f"{r.vencedor} ({f3(d['mae_vencedor'])})", f3(d["mae_vigente_antes"]),
                   d["decisao"].upper(), f"ciclo {list(hist['cycle_id']).index(d['vigente_apos']['cycle_id']) + 1}"])
    s += [Paragraph("5. Eleição do vencedor e promoção controlada", H1),
          Paragraph("O notebook 04 primeiro verifica a integridade do ciclo: os dois candidatos existem, pertencem ao mesmo "
                    "<i>cycle_id</i>/<i>data_version</i> e usaram as mesmas features. Qualquer falha interrompe o workflow. Depois:", P)]
    s += bullets([
        "<b>Vencedor do ciclo:</b> menor MAE na validação fixa (em caso de empate, vence o Ridge, que é mais simples).",
        "<b>Confronto com o vigente:</b> o modelo vigente é recarregado de <i>models/historico/</i>, com seu próprio pré-processamento e suas features, e reavaliado na <b>mesma validação fixa</b>.",
        "<b>Regra de promoção:</b> o desafiante precisa (a) ter MAE menor que o do baseline e (b) ser <b>pelo menos 2% melhor</b> que o vigente. Sem vigente, o primeiro modelo válido é promovido. Caso contrário, o vigente é <b>mantido</b>.",
        "O manifesto <i>models/modelo_vigente.json</i> guarda identificação, data_version, features, métrica, run_id, URI do modelo no MLflow, decisão, motivo, o modelo anterior e a última avaliação. Os modelos anteriores continuam preservados.",
    ])
    s += [tabela(lp, [1.2 * cm, 2.6 * cm, 3.8 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm]),
          Paragraph("Tabela 5 - Decisões automáticas por ciclo.", CAP),
          Paragraph("Motivos registrados: " + " ".join(f"<b>Ciclo {k}:</b> {d['motivo']}." for k, d in enumerate(decs, 1)), P),
          Paragraph(f"<b>Modelo vigente ao final:</b> {manifesto['algoritmo']}, promovido no ciclo "
                    f"{list(hist['cycle_id']).index(manifesto['cycle_id']) + 1} (data_version {manifesto['data_version']}), "
                    f"MAE de validação {manifesto['val_mae']:.3f} MPa, run_id <i>{manifesto['run_id']}</i>.", P)]

    # ------------------------------------------------------------ 6 MLflow
    s += [Paragraph("6. MLflow, persistência e rastreabilidade", H1),
          Paragraph("Experimento <i>n3-resistencia-concreto</i>, com backend SQLite (<i>mlflow.db</i>) e artefatos em <i>mlartifacts/</i>. "
                    "Cada ciclo gera cinco runs (etl, votacao, modeloA, modeloB, decisao) e mais um de previsoes, todos com as tags "
                    "<i>cycle_id</i>, <i>data_version</i> e <i>etapa</i>. São registrados os parâmetros do lote, o relatório de ETL e de drift, "
                    "os votos por feature e o método de votação, as features eleitas, os algoritmos, os hiperparâmetros, a seed, as métricas de "
                    "CV e de validação, o modelo serializado (com o código de <i>src/</i>), a decisão de promoção com o manifesto e as métricas "
                    "de teste do vigente. Os runs dos candidatos também recebem a tag <i>resultado</i> (promovido / venceu_mas_nao_promovido / perdeu).", P),
          Paragraph("<b>Persistência entre execuções.</b> Os runners do GitHub Actions são temporários. Por isso, ao final de cada ciclo bem-sucedido, "
                    "o workflow faz <b>commit</b> no próprio repositório de <i>mlflow.db</i>, <i>mlartifacts/</i>, <i>models/</i>, <i>evidencias/</i>, "
                    "do treino acumulado e do estado de ingestão, e também publica tudo como <b>artifact baixável</b>. O ciclo seguinte faz checkout "
                    "desse estado e recupera o manifesto e o modelo vigente sem depender do runner anterior. Como o MLflow grava caminhos absolutos, "
                    "<i>src/utils.py</i> reescreve automaticamente o prefixo dos artefatos para a pasta atual. O histórico pode ser consultado tanto "
                    "no runner quanto localmente com <i>mlflow ui --backend-store-uri sqlite:///mlflow.db</i>.", P)]

    # ------------------------------------------------------------ 7 arquitetura
    s += [Paragraph("7. Arquitetura do fluxo único", H1),
          Paragraph("Existe um único workflow, <i>.github/workflows/pipeline_mlops.yml</i>, com gatilho <i>schedule</i> "
                    "(cron <i>17,47 * * * *</i>) e <i>workflow_dispatch</i> para depuração. Ele tem um job sequencial, com "
                    "<i>concurrency</i> para impedir ciclos simultâneos. Os notebooks são executados em ordem com <i>jupyter nbconvert --execute</i>; "
                    "se qualquer etapa falhar, o job para antes do commit e nenhuma promoção é persistida.", P),
          tabela([
              ["#", "Etapa", "Onde", "Saída principal"],
              ["1", "Agendamento", "pipeline_mlops.yml (schedule)", "execução com gatilho schedule"],
              ["2", "Dados novos e versionamento", "01_etl · src/ingestao.py", "cycle_id, data_version, treino acumulado"],
              ["3", "ETL, validação, atributos e drift", "01_etl · src/preprocessamento.py, src/drift.py", "diagnósticos antes/depois, drift.csv"],
              ["4-5", "3 métodos + votação", "02_votacao_features · src/features.py", "votacao_features.csv, features eleitas"],
              ["6", "Treino A e B", "03_treino_competicao · src/modelagem.py", "2 runs MLflow, candidatos serializados"],
              ["7", "Eleição do vencedor", "04_selecao_promocao · src/promocao.py", "vencedor do ciclo"],
              ["8", "Registro MLflow", "todos os notebooks", "runs com cycle_id / data_version"],
              ["9", "Regra de promoção", "04_selecao_promocao", "modelo_vigente.json, decisao.json"],
              ["10", "Previsões + evidências", "05_previsoes · src/finalizar_ciclo.py", "previsoes_teste.csv, resumo, commit, artifact"],
          ], [1 * cm, 4.4 * cm, 5.8 * cm, 5.4 * cm]),
          Paragraph("Tabela 6 - Etapas do fluxo obrigatório e onde cada uma é implementada.", CAP)]

    # ------------------------------------------------------------ 8 evidências
    le = [["Ciclo", "cycle_id", "Gatilho", "Lote", "Decisão", "MAE teste (vigente)", "R<super>2</super> teste"]]
    for k, (r, c) in enumerate(zip(hist.itertuples(), ctxs), 1):
        le.append([k, r.cycle_id, r.gatilho, r.lote, r.decisao, f3(c["teste"]["mae"]), f3(c["teste"]["r2"])])
    gat_txt = ", ".join(f"{v} via <i>{k}</i>" for k, v in gatilhos.items())
    s += [Paragraph("8. Evidências de execução e previsões", H1),
          Paragraph(f"Foram executados <b>{len(hist)} ciclos</b> ({gat_txt}). O notebook 05 carrega o modelo vigente pelo manifesto, "
                    "junto com seu pré-processamento e suas features, e gera previsões para o <b>teste reservado</b>, que nunca foi usado "
                    "na escolha de modelos. Cada ciclo deixa em <i>evidencias/&lt;cycle_id&gt;/</i> os notebooks executados, as tabelas, "
                    "os gráficos, o <i>decisao.json</i> e um <i>resumo.md</i>, que também aparece no Summary da execução no GitHub.", P),
          tabela(le, [1.1 * cm, 4.9 * cm, 1.9 * cm, 2 * cm, 2.1 * cm, 2.6 * cm, 1.8 * cm]),
          Paragraph("Tabela 7 - Ciclos executados e desempenho do vigente no teste reservado.", CAP),
          Paragraph("A comparação entre ciclos mostra o que mudou a cada lote: a votação foi refeita, os dois algoritmos foram retreinados "
                    "e o vigente foi promovido sempre que a melhoria superou 2%. No ciclo sem dados novos, o desafiante ficou igual ao "
                    "vigente e o modelo foi <b>mantido</b> automaticamente.", P),
          figura(EV / "evolucao_ciclos.png", 14, "Figura 3 - Evolução do MAE de validação dos candidatos por ciclo."),
          figura(ciclos[-1] / "previsto_vs_real.png", 8.5, "Figura 4 - Previsto x real no teste reservado (modelo vigente final).")]
    prints = sorted((RAIZ / "reports/prints").glob("*.png")) if (RAIZ / "reports/prints").exists() else []
    if prints:
        s.append(Paragraph("Capturas de tela", H2))
        for n, pth in enumerate(prints, 5):
            legenda = pth.stem.split("_", 1)[-1].replace("_", " ")
            s.append(figura(pth, 15, f"Figura {n} - {legenda}."))

    # ------------------------------------------------------------ 9 critério 10
    s += [Paragraph("9. Recurso adicional: monitoramento de drift com alerta", H1),
          Paragraph("<b>Problema identificado.</b> O pipeline incorpora lotes automaticamente. Se um lote tiver perfil diferente do que "
                    "o modelo conhece, as métricas podem mudar sem que o grupo perceba o motivo, e a decisão de promoção passa a ser "
                    "tomada sobre dados que mudaram.", P),
          Paragraph("<b>Justificativa.</b> Nenhum critério de 1 a 9 exige observar a distribuição dos dados de entrada, mas na prática "
                    "essa é uma das primeiras causas de degradação de modelos em produção.", P),
          Paragraph("<b>Solução implementada.</b> <i>src/drift.py</i> aplica o teste de Kolmogorov-Smirnov, comparando cada variável do "
                    "lote novo (já limpo) com o treino acumulado até o ciclo anterior (alpha = 0,05). Se 25% ou mais das variáveis "
                    "apresentarem drift, é emitido um <b>alerta</b>.", P),
          Paragraph("<b>Ponto de integração.</b> Notebook 01, logo após a limpeza e antes da incorporação. O resultado vai para o MLflow "
                    "(métricas <i>drift_variaveis</i> e <i>drift_alerta</i> e artefatos), para <i>evidencias/&lt;cycle_id&gt;/drift.csv/.png</i>, "
                    "para o histórico de ciclos e para o GitHub Actions, como anotação <i>warning</i> e destaque no Summary da execução.", P)]
    if i_drift is not None:
        dr = pd.read_csv(ciclos[i_drift] / "drift.csv")
        ld = [["Variável", "KS", "p-valor", "Média treino", "Média lote", "Drift"]]
        for _, r in dr.iterrows():
            ld.append([r["variavel"], f"{r['ks_estatistica']:.3f}", f"{r['p_valor']:.2e}", f"{r['media_referencia']:.2f}", f"{r['media_lote']:.2f}", "sim" if r["drift"] else "não"])
        sem_alerta = [str(k + 1) for k in com_lote if k != i_drift and ctxs[k].get("drift_resumo")]
        s += [Paragraph(f"<b>Resultado demonstrado.</b> Nos ciclos {', '.join(sem_alerta)}, os lotes aleatórios não apresentaram drift. "
                        f"No ciclo {i_drift + 1} ({ctxs[i_drift]['lote']}), o alerta foi disparado: {ctxs[i_drift]['drift_resumo']}. "
                        "A idade média do lote foi muito maior que a do treino, como esperado pela simulação. O alerta explica a queda "
                        "acentuada do MAE do Ridge nesse ciclo e sinaliza que o modelo passou a ver uma região do espaço de entrada que "
                        "antes não conhecia.", P),
              tabela(ld, [4 * cm, 1.8 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 1.6 * cm]),
              Paragraph(f"Tabela 8 - Teste KS do ciclo {i_drift + 1}.", CAP)]
        if (ciclos[i_drift] / "drift.png").exists():
            s.append(figura(ciclos[i_drift] / "drift.png", 12, f"Figura - Estatística KS por variável no ciclo {i_drift + 1} (vermelho = drift)."))

    # ------------------------------------------------------------ 10 limitações / conclusões
    s += [Paragraph("10. Limitações", H1)]
    s += bullets([
        "O dataset é pequeno e de laboratório, e a chegada de dados é <b>simulada</b> por lotes, sem ordem temporal real.",
        "A votação é feita uma vez sobre todo o treino acumulado. A CV de 5 folds no treino usa esse conjunto já eleito e por isso tende a ser levemente otimista; por esse motivo, as decisões usam a validação fixa, que é independente.",
        "A validação fixa tem 151 amostras. Diferenças pequenas de MAE estão sujeitas a ruído, e é por isso que existe o limiar de 2%.",
        "Os hiperparâmetros são fixos (não há busca), o que deixa o pipeline simples e rápido, mas provavelmente abaixo do desempenho ótimo.",
        "Versionar o estado do MLflow no repositório é simples e gratuito, mas o repositório cresce cerca de 5 MB por ciclo. Por isso, após a demonstração, o agendamento deve passar a ser semanal.",
        "O teste KS só detecta mudança nas distribuições marginais, não em relações entre variáveis.",
    ])
    s += [Paragraph("11. Conclusões", H1),
          Paragraph("O projeto entrega um pipeline MLOps funcional, reprodutível e versionado. Um único workflow agendado incorpora dados "
                    "novos, faz ETL sem vazamento, elege features pela votação de três métodos, coloca dois algoritmos para competir com o "
                    "mesmo protocolo, registra tudo no MLflow com identificadores de ciclo e decide automaticamente entre promover e manter "
                    "o modelo vigente. Os resultados mostram o comportamento esperado: a cada lote novo o erro diminuiu e o modelo foi promovido "
                    f"(MAE de teste de {ctxs[0]['teste']['mae']:.2f} para {ctxs[-1]['teste']['mae']:.2f} MPa), e sem dados novos o vigente foi "
                    "preservado. O monitoramento de drift complementa o fluxo ao explicar mudanças de desempenho que, de outra forma, passariam "
                    "despercebidas.", P),
          Paragraph("Referências", H2),
          Paragraph("YEH, I-C. Modeling of strength of high-performance concrete using artificial neural networks. <i>Cement and Concrete "
                    "Research</i>, v. 28, n. 12, p. 1797-1808, 1998.", P),
          Paragraph("UCI Machine Learning Repository. Concrete Compressive Strength. https://archive.ics.uci.edu/dataset/165", P),
          Paragraph("PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. JMLR, v. 12, p. 2825-2830, 2011. · MLflow Documentation. https://mlflow.org/docs", P)]

    def rodape(canvas, doc_):
        if doc_.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(2.2 * cm, 1.2 * cm, "Católica SC · Análise Preditiva · Trabalho N3")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc_.page}")
        canvas.restoreState()

    doc.build(s, onFirstPage=rodape, onLaterPages=rodape)
    print("Relatório gerado em reports/relatorio_n3.pdf")


if __name__ == "__main__":
    main()
