"""Página inicial: o projeto em 30 segundos, para quem chega pelo link."""

from __future__ import annotations

import streamlit as st

from src.business import fmt_brl, fmt_int, fmt_num, fmt_pct
from src.labels import model_with_article
from src.ui import components
from src.ui.data import metrics
from src.ui.html import esc, ui_html

STACK = [
    "Python",
    "scikit-learn",
    "FastAPI",
    "Streamlit",
    "LangChain",
    "Gemini",
    "ChromaDB",
    "Docker",
    "MLflow",
    "GitHub Actions",
]

EXPLORE = [
    (
        "cliente",
        "Cliente",
        ":material/person_search:",
        "Monte um perfil e veja o risco, a recomendação, o que pesou na previsão e ofertas que "
        "mudariam o risco.",
    ),
    (
        "carteira",
        "Carteira",
        ":material/groups:",
        "Envie um CSV com vários clientes: validação linha a linha, fila de contato e monitor de "
        "mudança no perfil da base.",
    ),
    (
        "estrategia",
        "Estratégia",
        ":material/payments:",
        "Ajuste o custo da oferta e o valor do cliente e veja o novo ponto de corte e o retorno "
        "da campanha.",
    ),
    (
        "modelo",
        "Modelo",
        ":material/fact_check:",
        "Como o modelo foi escolhido, calibração, o que ele aprendeu, onde erra mais e seus "
        "limites.",
    ),
    (
        "assistente",
        "Assistente",
        ":material/smart_toy:",
        "Pergunte em português: o agente consulta o modelo (function calling) e a base de "
        "conhecimento (RAG).",
    ),
]


# Radar decorativo do hero: anéis, varredura e alguns clientes ("blips")
# coloridos pela faixa de risco — o nome do produto em forma de imagem. Some
# abaixo de 980 px (CSS) e é ignorado por leitores de tela.
_RADAR_RINGS = "".join(
    f'<circle cx="140" cy="140" r="{r}" fill="none" stroke="rgba(34,211,238,{a})" />'
    for r, a in ((46, 0.20), (92, 0.16), (138, 0.12))
)
_RADAR_BLIPS = (
    ("alto", 68, 30),
    ("alto", 79, 58),
    ("medio", 30, 38),
    ("medio", 61, 74),
    ("baixo", 24, 66),
    ("baixo", 44, 82),
    ("baixo", 39, 22),
    ("baixo", 82, 40),
)
RADAR = (
    '<div class="cr-radar" aria-hidden="true">'
    '<svg viewBox="0 0 280 280">'
    f"{_RADAR_RINGS}"
    '<line x1="0" y1="140" x2="280" y2="140" stroke="rgba(34,211,238,0.12)" />'
    '<line x1="140" y1="0" x2="140" y2="280" stroke="rgba(34,211,238,0.12)" />'
    "</svg>"
    '<div class="cr-radar-sweep"></div>'
    + "".join(
        f'<i class="cr-blip cr-blip--{kind}" style="left:{x}%;top:{y}%"></i>'
        for kind, x, y in _RADAR_BLIPS
    )
    + '<span class="cr-radar-tag" style="left:68%;top:30%">▲ risco alto</span>'
    "</div>"
)


def _hero() -> None:
    stack = "".join(f"<span>{esc(item)}</span>" for item in STACK)
    ui_html(
        '<div class="cr-hero"><div class="cr-hero-grid"><div>'
        '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center">'
        '<p class="cr-eyebrow" style="margin:0">Machine learning de ponta a ponta</p>'
        f"{components.status_chip()}</div>"
        '<div class="cr-hero-title" role="heading" aria-level="1" style="margin-top:14px">Churn Radar</div>'
        '<p class="cr-hero-text">Descobre quais clientes de uma operadora de telecom estão perto '
        "de cancelar, explica o porquê de cada previsão e calcula se vale a pena agir — com um "
        "modelo calibrado, uma API publicada e um assistente de IA que consulta o modelo e uma "
        "base de conhecimento.</p>"
        f'<div class="cr-stack">{stack}</div>'
        f"</div>{RADAR}</div></div>"
    )
    from src.ui.nav import PAGES

    st.write("")
    with st.container(horizontal=True, gap="small", key="cta"):
        st.page_link(PAGES["cliente"], label="Analisar um cliente", icon=":material/arrow_forward:")
        st.page_link(
            PAGES["modelo"], label="Ver como o modelo foi escolhido", icon=":material/fact_check:"
        )
        st.page_link(
            PAGES["assistente"], label="Conversar com o assistente", icon=":material/smart_toy:"
        )


def _numbers() -> None:
    m = metrics()
    if not m:
        return
    op, test, ci = m["operational"], m["test_metrics"], m["ci95"]
    components.section(
        "Resultados",
        "Em clientes que o modelo nunca viu",
        f"Avaliação única em {fmt_int(m['n_test'])} clientes separados antes do treino.",
    )
    components.kpis(
        [
            {
                "label": "Cancelamentos encontrados",
                "value": fmt_pct(op["recall"], 0),
                "note": f"contatando {fmt_pct(op['contact_rate'], 0)} da base — a outra metade "
                "da lista seria desperdício sem o modelo",
                "accent": True,
            },
            {
                "label": "ROC-AUC",
                "value": fmt_num(test["roc_auc"], 2),
                "note": f"ordena bem quem cancela antes de quem fica (intervalo de 95%: "
                f"{fmt_num(ci['roc_auc'][0], 2)} a {fmt_num(ci['roc_auc'][1], 2)})",
            },
            {
                "label": "Erro de calibração",
                "value": f"{fmt_num(test['ece'] * 100, 1)} pp",
                "note": "a probabilidade prevista fica perto da taxa real de cancelamento",
            },
            {
                "label": "Valor por 1.000 clientes",
                "value": fmt_brl(op["net_value_per_1000"]),
                "note": "retorno líquido estimado da campanha, com hipóteses de custo ajustáveis",
            },
        ]
    )


def _how() -> None:
    m = metrics()
    components.section(
        "Como funciona",
        "Do dado bruto ao produto",
        "Cada etapa foi pensada para o número na tela ser confiável — e para o projeto rodar de "
        "graça.",
    )
    # Números e nomes vêm de models/metrics.json: um retreino que escolha
    # outro modelo atualiza este texto sozinho.
    n = m.get("n_samples")
    clients = f"{fmt_int(n)} clientes reais" if n else "Clientes reais"
    chosen = m.get("model_selected")
    winner = f": {model_with_article(chosen)}" if chosen else ""
    steps = [
        (
            "Dados",
            f"{clients} de uma operadora (dataset público da IBM). 20% ficam de fora "
            "como teste e só são usados na avaliação final.",
        ),
        (
            "Modelo",
            "Quatro candidatos comparados com validação cruzada 5×3. Vence o mais simples dentro "
            f"do empate técnico{winner}.",
        ),
        (
            "Decisão",
            "O corte de contato sai de uma conta de valor — oferta × chance de sucesso × valor do "
            "cliente —, não do 0,5 padrão.",
        ),
        (
            "Produto",
            "API <code>FastAPI</code> no Render, interface em Streamlit e um assistente com "
            "Gemini que usa o modelo por function calling e a documentação por RAG.",
        ),
    ]
    cards = "".join(
        f'<div class="cr-step"><div class="cr-step-n">Etapa {i}</div>'
        f'<div class="cr-step-t">{esc(title)}</div><div class="cr-step-d">{text}</div></div>'
        for i, (title, text) in enumerate(steps, start=1)
    )
    ui_html(f'<div class="cr-steps">{cards}</div>')


def _selection_text(m: dict) -> str:
    """Resumo da escolha do modelo a partir de metrics.json (regra de 1 erro-padrão)."""
    selection = m.get("selection", {})
    best, chosen = selection.get("best_mean"), selection.get("chosen")
    if not chosen:
        return "Entre modelos empatados dentro do ruído da validação cruzada, fica o mais simples."
    gain = (
        "explicação exata e uma API mais leve"
        if chosen == "logistic_regression"
        else "o mais simples entre os empatados"
    )
    if best and best != chosen:
        return (
            f"{model_with_article(best, capitalize=True)} teve a maior média, mas a diferença "
            f"ficou abaixo de 1 erro-padrão. {model_with_article(chosen, capitalize=True)} "
            f"ganhou: {gain}."
        )
    return f"{model_with_article(chosen, capitalize=True)} teve a maior média e é o modelo em produção."


def _decisions() -> None:
    m = metrics()
    r2 = m.get("redundancy", {}).get("monthly_r2_services")
    r2_text = f" (R² = {fmt_num(r2, 3)})" if r2 is not None else ""
    components.section(
        "Decisões de projeto",
        "O que diferencia este modelo",
        "Escolhas que um projeto de churn costuma pular — e o que cada uma resolve.",
    )
    cards = [
        (
            "Probabilidades em que dá para confiar",
            "Sem reponderação de classes, que inflava o risco médio da base. Um cliente com 30% de "
            "risco é, de fato, um cliente em que 3 de cada 10 cancelam.",
        ),
        ("O mais simples dentro do ruído", _selection_text(m)),
        (
            "Variáveis que não enganam",
            f"A mensalidade é determinada pelos serviços contratados{r2_text}. Fora do modelo, "
            "os pesos voltam a fazer sentido de negócio, sem perder qualidade.",
        ),
        (
            "Sem tela de espera",
            "A API gratuita hiberna depois de 15 minutos. Enquanto ela acorda, o app calcula a "
            "mesma previsão aqui, com o mesmo modelo, e avisa de onde veio o número.",
        ),
    ]
    html = "".join(
        f'<div class="cr-card"><div class="cr-reco-title" style="margin-bottom:6px">{esc(t)}</div>'
        f'<div class="cr-text" style="font-size:13.5px">{d}</div></div>'
        for t, d in cards
    )
    ui_html(f'<div class="cr-decisions">{html}</div>')


def _explore() -> None:
    from src.ui.nav import PAGES

    components.section("Explore", "O que dá para fazer aqui")
    cols = st.columns(len(EXPLORE), gap="small")
    for col, (path, title, icon, text) in zip(cols, EXPLORE, strict=True):
        with col, st.container(border=True, height="stretch"):
            ui_html(
                f'<div class="cr-explore-t">{esc(title)}</div>'
                f'<div class="cr-step-d" style="min-height:106px">{esc(text)}</div>'
            )
            st.page_link(PAGES[path], label=f"Abrir {title.lower()}", icon=icon)


def render() -> None:
    _hero()
    _numbers()
    _how()
    _decisions()
    _explore()
