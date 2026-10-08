"""Página "Estratégia": simulador de retorno da campanha de retenção."""

from __future__ import annotations

import math

import streamlit as st

from src.business import (
    DEFAULT_LTV,
    DEFAULT_OFFER_COST,
    DEFAULT_SUCCESS_RATE,
    fmt_brl,
    fmt_int,
    fmt_pct,
    optimal_threshold,
    theoretical_threshold,
    value_curve,
)
from src.ui import charts, components
from src.ui.data import COST_KEYS, campaign_policy, evaluation, reference_costs
from src.ui.html import esc, ui_html
from src.ui.theme import ACCENT, ACCENT_2, MUTED


def brl_short(value: float) -> str:
    """R$ compacto para eixos: R$ 217 mil, R$ 1,2 mi."""
    sign = "−" if value < 0 else ""
    v = abs(value)
    if v >= 1_000_000:
        text = f"{v / 1_000_000:.1f}".replace(".", ",") + " mi"
    elif v >= 1_000:
        text = f"{v / 1_000:.0f} mil"
    else:
        text = f"{v:.0f}"
    return f"{sign}R$ {text}"


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Marcas "redondas" que cobrem [lo, hi] (passos de 1, 2, 2,5 ou 5 × 10^k)."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / (n - 1)
    base = 10 ** math.floor(math.log10(raw))
    step = next(m * base for m in (1, 2, 2.5, 5, 10) if m * base >= raw)
    start = math.floor(lo / step) * step
    count = int(round((math.ceil(hi / step) * step - start) / step))
    return [start + i * step for i in range(count + 1)]


PERSIST = {"persist_state": "session"}


def _reference_state() -> dict:
    ref = reference_costs()
    return {
        COST_KEYS["ltv"]: int(ref["ltv"]),
        COST_KEYS["offer_cost"]: int(ref["offer_cost"]),
        COST_KEYS["success_pct"]: int(round(ref["success_rate"] * 100)),
    }


def _reset_costs() -> None:
    st.session_state.update(_reference_state())


def _controls() -> tuple[float, float, float, int]:
    """Controles das hipóteses. Os valores ficam na sessão (persist_state)
    e valem também para a página Carteira — ver data.campaign_policy."""
    for key, value in _reference_state().items():
        st.session_state.setdefault(key, value)
    with st.container(key="panel"):
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        ltv = c1.slider(
            "Valor do cliente (R$)",
            200,
            5000,
            step=100,
            key=COST_KEYS["ltv"],
            help="Quanto a empresa preserva quando um cliente que ia cancelar fica.",
            **PERSIST,
        )
        offer = c2.slider(
            "Custo da oferta (R$)",
            10,
            500,
            step=10,
            key=COST_KEYS["offer_cost"],
            help="Desconto, brinde ou tempo do time de retenção, por cliente contatado.",
            **PERSIST,
        )
        success = c3.slider(
            "Chance de a oferta funcionar",
            5,
            80,
            step=5,
            format="%d%%",
            key=COST_KEYS["success_pct"],
            help="Fração dos clientes que iam cancelar e ficam depois do contato.",
            **PERSIST,
        )
        base = c4.number_input(
            "Clientes na carteira", min_value=1_000, max_value=5_000_000, value=10_000, step=1_000
        )
    if campaign_policy()["custom"]:
        st.button(
            "Voltar às hipóteses de referência",
            icon=":material/restart_alt:",
            on_click=_reset_costs,
            help="As hipóteses ajustadas aqui também valem na página Carteira.",
        )
    return float(ltv), float(offer), success / 100, int(base)


def render() -> None:
    ev = evaluation()
    components.page_header(
        "Estratégia de retenção",
        "Vale a pena contatar quem?",
        "Ajuste as hipóteses da campanha e veja o corte que maximiza o retorno. A conta usa os "
        "clientes de teste, que o modelo não viu no treino. As hipóteses ajustadas aqui também "
        "valem na página Carteira.",
    )
    if not ev.get("curves"):
        components.note(
            "Rode <code>python -m src.train</code> para gerar as curvas de avaliação.", "warn"
        )
        return

    ltv, offer, success, base = _controls()

    # O corte é escolhido nas previsões fora da amostra do TREINO (como no
    # treino de verdade) e o resultado é medido no TESTE — escolher e medir no
    # mesmo conjunto superestimaria o retorno.
    best_t = optimal_threshold(ev["curves"]["oof"], ltv, offer, success)
    curve = value_curve(ev["curves"]["test"], ltv, offer, success)
    n_test = curve[0]["tp"] + curve[0]["fp"] + curve[0]["fn"] + curve[0]["tn"]
    scale = base / n_test
    best = min(curve, key=lambda p: abs(p["threshold"] - best_t))
    peak = max(curve, key=lambda p: (p["valor"], p["threshold"]))
    theory = theoretical_threshold(ltv, offer, success)
    production_t = float(ev["optimal_threshold"])
    production = min(curve, key=lambda p: abs(p["threshold"] - production_t))
    everyone = curve[0]

    components.kpis(
        [
            {
                "label": "Corte recomendado",
                "value": fmt_pct(best_t, 0),
                "note": f"escolhido nos dados de treino; pela conta teórica, {fmt_pct(theory, 0)}",
                "accent": True,
            },
            {
                "label": "Retorno líquido",
                "value": fmt_brl(best["valor"] * scale),
                "note": f"para {fmt_int(base)} clientes, contra não fazer nada",
            },
            {
                "label": "Clientes contatados",
                "value": fmt_pct(best["taxa_contato"], 0),
                "note": f"e {fmt_pct(best['recall'], 0)} dos que iam cancelar entram na lista",
            },
            {
                "label": "Contra contatar todos",
                "value": ("+" if best["valor"] >= everyone["valor"] else "")
                + fmt_brl((best["valor"] - everyone["valor"]) * scale),
                "note": f"contatar a carteira inteira daria {fmt_brl(everyone['valor'] * scale)}",
            },
        ]
    )

    # ---- Curva de valor ------------------------------------------------------
    components.section(
        "Retorno × corte",
        "Quanto a campanha rende em cada ponto de corte",
        "Cortes baixos gastam ofertas com quem ia ficar; cortes altos deixam de fora clientes "
        "que iam cancelar. O retorno é medido nos clientes de teste.",
    )
    values = [p["valor"] * scale for p in curve]
    lo, hi = min(0.0, min(values)), max(values)
    pad = (hi - lo) * 0.08 or 1.0
    y_ticks = nice_ticks(lo, hi + pad)
    hover = [
        (
            p["threshold"] * 100,
            f"<b>corte {fmt_pct(p['threshold'], 0)}</b><br>retorno {esc(fmt_brl(p['valor'] * scale))}"
            f"<br>contata {fmt_pct(p['taxa_contato'], 0)} · encontra {fmt_pct(p['recall'], 0)}",
        )
        for p in curve
    ]
    chart = charts.line_chart(
        series=[
            {
                "points": [(p["threshold"] * 100, p["valor"] * scale) for p in curve],
                "color": ACCENT,
                "fill": True,
            },
            {"points": [(0, 0), (100, 0)], "color": MUTED, "width": 1},
        ],
        x_domain=(0, 100),
        y_domain=(min(y_ticks), max(y_ticks)),
        x_ticks=[(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")],
        y_ticks=[(v, brl_short(v)) for v in y_ticks],
        hover=hover,
        markers=[
            {
                "x": best_t * 100,
                "y": best["valor"] * scale,
                "color": ACCENT,
                "label": f"Corte recomendado: {fmt_pct(best_t, 0)}",
            }
        ],
        vlines=[
            {
                "x": production_t * 100,
                "label": f"em produção: {fmt_pct(production_t, 0)}",
                "color": ACCENT_2,
            },
        ],
        x_title="corte de probabilidade para contatar",
        aria="Retorno líquido da campanha para cada corte de probabilidade",
    )
    components.card(
        charts.legend(
            [("Retorno líquido", ACCENT, "line"), ("Corte em produção", ACCENT_2, "line")]
        )
        + chart
    )
    if abs(peak["threshold"] - best_t) >= 0.02 and peak["valor"] > 0:
        gap = (peak["valor"] - best["valor"]) / peak["valor"]
        components.note(
            f"Nos clientes de teste, o retorno máximo apareceu no corte de "
            f"{fmt_pct(peak['threshold'], 0)} — só {fmt_pct(gap, 0)} acima do corte recomendado. "
            "A curva é plana perto do topo, e escolher o corte olhando o próprio teste seria "
            "otimista demais: por isso ele é escolhido nos dados de treino."
        )

    # ---- Curva de ganho --------------------------------------------------------
    components.section(
        "Eficiência da lista",
        "Quantos cancelamentos a lista encontra",
        "Com os clientes em ordem de risco, a curva mostra que fração dos cancelamentos aparece "
        "conforme a lista cresce. A diagonal é uma lista em ordem aleatória.",
    )
    pts = sorted({(p["taxa_contato"] * 100, p["recall"] * 100) for p in curve})
    gain = charts.line_chart(
        series=[
            {"points": [(0, 0), (100, 100)], "color": MUTED, "width": 1},
            {"points": pts, "color": ACCENT},
        ],
        x_domain=(0, 100),
        y_domain=(0, 100),
        x_ticks=[(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")],
        y_ticks=[(v, f"{v}%") for v in (0, 25, 50, 75, 100)],
        markers=[
            {
                "x": best["taxa_contato"] * 100,
                "y": best["recall"] * 100,
                "color": ACCENT,
                "label": f"Corte recomendado: contata {fmt_pct(best['taxa_contato'], 0)} e encontra "
                f"{fmt_pct(best['recall'], 0)}",
            }
        ],
        hover=[
            (x, f"<b>contata {x:.0f}%</b><br>encontra {y:.0f}% dos cancelamentos".replace(".", ","))
            for x, y in pts[:: max(len(pts) // 50, 1)]
        ],
        x_title="parte da carteira contatada",
        height=240,
        aria="Curva de ganho acumulado do modelo contra uma lista aleatória",
    )
    components.card(
        charts.legend([("Modelo", ACCENT, "line"), ("Lista aleatória", MUTED, "line")]) + gain
    )

    # ---- Comparação --------------------------------------------------------------
    components.section("Comparação", "Formas de montar a campanha")
    rows = [("Não fazer nada", None), ("Contatar todos", everyone)]
    if production is best:
        rows.append(
            (f"Modelo, corte recomendado ({fmt_pct(best_t, 0)}, o mesmo em produção)", best)
        )
    else:
        rows.append((f"Modelo, corte em produção ({fmt_pct(production_t, 0)})", production))
        rows.append((f"Modelo, corte recomendado ({fmt_pct(best_t, 0)})", best))
    body = []
    for label, p in rows:
        if p is None:
            cells = ["0", "0", fmt_brl(0), fmt_brl(0), fmt_brl(0)]
        else:
            saved = p["tp"] * success * scale
            cost = p["contatados"] * offer * scale
            cells = [
                fmt_int(p["contatados"] * scale),
                fmt_int(saved),
                fmt_brl(cost),
                fmt_brl(saved * ltv),
                fmt_brl(p["valor"] * scale),
            ]
        hl = ' class="cr-hl"' if p is best else ""
        body.append(
            f"<tr{hl}><td>{esc(label)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        )
    ui_html(
        '<div class="cr-table-wrap"><table class="cr-table"><thead><tr>'
        "<th>Estratégia</th><th>Contatados</th><th>Clientes salvos</th><th>Custo das ofertas</th>"
        "<th>Receita preservada</th><th>Retorno líquido</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )
    components.note(
        "<b>Como a conta é feita:</b> retorno líquido = clientes que iam cancelar e foram "
        "contatados × chance de a oferta funcionar × valor do cliente − clientes contatados × "
        "custo da oferta. Quem não é contatado não entra na conta, porque não fazer nada também "
        "o perde. As hipóteses de referência "
        f"({fmt_brl(DEFAULT_LTV)}, {fmt_brl(DEFAULT_OFFER_COST)} e {fmt_pct(DEFAULT_SUCCESS_RATE, 0)}) "
        "são ilustrativas, não números de uma empresa real."
    )
