"""Demonstração interativa. Rode com:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st

from src.prever import PADRAO, carregar_modelo, prever

st.set_page_config(page_title="Resistência do Concreto", page_icon="🧱", layout="wide")
st.title("🧱 Previsão da resistência à compressão do concreto")
with st.spinner("Preparando o modelo de previsão…"):
    pacote = carregar_modelo()
faixas = pacote["faixas"]

st.caption(f"Modelo: {pacote['nome']} · erro médio no teste: "
           f"{pacote['metricas_teste']['MAE']:.1f} MPa · R² {pacote['metricas_teste']['R2']:.2f}")

ROTULOS = {
    "cimento": "Cimento (kg/m³)", "escoria": "Escória de alto-forno (kg/m³)",
    "cinza_volante": "Cinza volante (kg/m³)", "agua": "Água (kg/m³)",
    "superplastificante": "Superplastificante (kg/m³)",
    "agregado_graudo": "Agregado graúdo / brita (kg/m³)",
    "agregado_miudo": "Agregado miúdo / areia (kg/m³)",
}

with st.sidebar:
    st.header("Composição da mistura")
    entradas: dict[str, float] = {}
    for col, rotulo in ROTULOS.items():
        mn, mx = faixas[col]
        entradas[col] = st.slider(rotulo, float(mn), float(mx), float(PADRAO[col]), step=1.0)
    idade_min, idade_max = faixas["idade"]
    entradas["idade"] = st.slider("Idade do concreto (dias)", int(idade_min), int(idade_max), 28)

r = prever(**entradas)
c1, c2, c3 = st.columns(3)
c1.metric("Resistência estimada", f"{r['resistencia_MPa']:.1f} MPa")
c2.metric("Faixa empírica de referência", f"{r['intervalo_80'][0]:.1f} – {r['intervalo_80'][1]:.1f} MPa")
c3.metric("Relação água/cimento", f"{entradas['agua'] / entradas['cimento']:.2f}")
for aviso in r["avisos"]:
    st.warning(aviso)

st.subheader("Resistência estimada ao longo do tempo")
dias = np.unique(np.geomspace(idade_min, idade_max, 40).round().astype(int))
X = pd.DataFrame([{**entradas, "idade": d} for d in dias])[pacote["entradas"]]
curva = pd.DataFrame({"Idade (dias)": dias,
                      "Resistência estimada (MPa)": pacote["modelo"].predict(X)}).set_index("Idade (dias)")
st.line_chart(curva)
st.caption("A curva mostra as estimativas do modelo para a composição selecionada.")

with st.expander("Sobre o modelo"):
    st.markdown(
        "- **Dados:** Concrete Compressive Strength (UCI), 1.030 ensaios de laboratório.\n"
        "- **Validação:** as métricas exibidas vêm do teste com traços inéditos. Após a avaliação, "
        "o modelo da demonstração é ajustado com todos os ensaios disponíveis.\n"
        f"- **Faixa empírica:** margem de {pacote['margem_80']:.1f} MPa, estimada na validação cruzada. "
        f"A cobertura observada no teste foi de {pacote['cobertura_intervalo']:.0%}.\n"
        "- **Limitação:** não substitui o ensaio de rompimento de corpos de prova; serve para "
        "estimar e comparar traços antes do ensaio.")
