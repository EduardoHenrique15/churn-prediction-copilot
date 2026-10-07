"""Páginas do app (criadas uma vez por processo, para st.page_link encontrar
exatamente os mesmos objetos que st.navigation recebeu)."""

from __future__ import annotations

import streamlit as st

from src.ui.pages import assistente, carteira, cliente, estrategia, inicio, modelo

PAGES: dict[str, st.Page] = {
    "inicio": st.Page(
        inicio.render, title="Visão geral", icon=":material/radar:", url_path="inicio", default=True
    ),
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