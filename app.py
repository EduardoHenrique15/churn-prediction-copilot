"""Churn Radar — interface Streamlit.

Rode a partir da raiz do projeto: `streamlit run app.py`

Este arquivo só monta a moldura (tema, navegação e rodapé). Cada página
mora em src/ui/pages/, os gráficos em src/ui/charts.py e o HTML seguro em
src/ui/html.py.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui import components
from src.ui.nav import PAGES
from src.ui.theme import STYLES

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="Churn Radar — previsão de cancelamento de clientes",
    page_icon=str(ASSETS / "favicon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(STYLES)
st.logo(str(ASSETS / "logo.svg"), size="large")

page = st.navigation(list(PAGES.values()), position="top")
# Título da aba do navegador por página ("Cliente · Churn Radar").
if page.url_path not in ("", "inicio"):
    st.set_page_config(page_title=f"{page.title} · Churn Radar")
page.run()
components.footer()
