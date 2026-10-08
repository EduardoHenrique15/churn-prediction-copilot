"""Páginas do app (criadas uma vez por processo, para st.page_link encontrar
exatamente os mesmos objetos que st.navigation recebeu)."""

from __future__ import annotations

import streamlit as st

from src.ui.pages import assistente, carteira, cliente, estrategia, inicio, modelo


def _to_home() -> None:
    st.switch_page(PAGES["inicio"])


PAGES: dict[str, st.Page] = {
    # A página padrão mora em "/" (o Streamlit ignora o url_path dela).
    "inicio": st.Page(inicio.render, title="Visão geral", icon=":material/radar:", default=True),
    # "/inicio" redireciona para "/" em vez de abrir o aviso "Page not found".
    "inicio_alias": st.Page(_to_home, title="Visão geral", url_path="inicio", visibility="hidden"),
    "cliente": st.Page(
        cliente.render, title="Cliente", icon=":material/person_search:", url_path="cliente"
    ),
    "carteira": st.Page(
        carteira.render, title="Carteira", icon=":material/groups:", url_path="carteira"
    ),
    "estrategia": st.Page(
        estrategia.render, title="Estratégia", icon=":material/payments:", url_path="estrategia"
    ),
    "modelo": st.Page(
        modelo.render, title="Modelo", icon=":material/fact_check:", url_path="modelo"
    ),
    "assistente": st.Page(
        assistente.render, title="Assistente", icon=":material/smart_toy:", url_path="assistente"
    ),
}
