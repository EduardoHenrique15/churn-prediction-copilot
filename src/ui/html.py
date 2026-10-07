"""HTML seguro para o Streamlit.

`st.markdown(..., unsafe_allow_html=True)` passa a string por um parser de
Markdown ANTES de renderizar o HTML. Duas armadilhas fazem o markup aparecer
como texto cru na tela:

1. qualquer linha com 4+ espaços de indentação vira bloco de código;
2. uma linha em branco no meio do HTML quebra o bloco em dois.

Todo HTML dinâmico da interface passa por `ui_html()`, que remove a
indentação de cada linha e junta tudo SEM espaço — as duas armadilhas deixam
de existir por construção. A consequência: dentro de HTML, nunca quebre uma
frase no meio de uma linha (as palavras colariam). Monte o texto com
literais concatenados, cada frase inteira no mesmo literal.

O CSS fica em `theme.STYLES`, que NÃO é f-string: as chaves do CSS brigam com
a interpolação e essa é a outra origem clássica do mesmo bug.
"""

from __future__ import annotations

import html

import streamlit as st


def ui_html(markup: str) -> None:
    """Renderiza HTML imune ao parser de Markdown (ver nota do módulo)."""
    compact = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


def esc(value) -> str:
    """Escapa texto que vem de fora (CSV, respostas do modelo) para HTML."""
    return html.escape(str(value), quote=True)


def md_text(text: str) -> str:
    """Texto seguro para st.markdown/st.caption.

    O Markdown do Streamlit interpreta `$...$` como fórmula: "R$ 100 e R$ 200"
    viraria LaTeX entre os dois cifrões. Escapar o cifrão resolve.
    """
    return str(text).replace("$", "\\$")