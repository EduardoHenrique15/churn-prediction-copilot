"""Página "Modelo": o model card — escolha, qualidade, calibração, o que o
modelo aprendeu, onde erra mais e para que ele não serve."""

from __future__ import annotations

import math
from datetime import datetime

import streamlit as st

from src.business import fmt_int, fmt_num, fmt_pct
from src.labels import FIELD_LABELS, value_label
from src.ui import charts, components
from src.ui.data import evaluation, metrics
from src.ui.html import esc, ui_html
from src.ui.theme import ACCENT, MUTED

MODEL_NAMES = {
    "dummy_baseline": "Baseline (taxa média)",
    "logistic_regression": "Regressão logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
}
COMPLEXITY = {
    "dummy_baseline": "não aprende nada — prevê a taxa média para todos",
    "logistic_regression": "28 pesos, explicação exata",
    "random_forest": "300 árvores",
    "hist_gradient_boosting": "árvores em sequência",
}
SUBGROUPS = [
    ("senior_citizen", "Idosos"),
    ("non_senior_citizen", "Não idosos"),
    ("contract_month_to_month", "Contrato mensal"),
    ("contract_one_year", "Contrato anual"),
    ("contract_two_year", "Contrato bienal"),
]


def _date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
    except ValueError:
        return iso


def _meta(m: dict) -> None:
    chips = [
        MODEL_NAMES.get(m.get("model_selected"), m.get("model_selected", "—")),
        f"treinado em {_date(m.get('trained_at'))}",
        f"versão {m.get('model_version', '—')}",
        f"{m.get('n_features', '—')} variáveis",
        f"scikit-learn {m.get('scikit_learn', '—')}",
    ]
    if m.get("git_sha"):
        chips.append(f"commit {m['git_sha']}")
    ui_html(
        '<div class="cr-stack" style="margin:-6px 0 6px 0">'
        + "".join(f"<span>{esc(c)}</span>" for c in chips)
        + "</div>"
    )


def _test_results(m: dict) -> None:
    test, ci, cv = m["test_metrics"], m["ci95"], m["cv_comparison"]
    baseline_brier = cv.get("dummy_baseline", {}).get("oof_brier")
    components.section(
        "Resultado no teste",
        "Quão bom é o modelo",
        f"Medido uma única vez em {fmt_int(m['n_test'])} clientes separados antes de qualquer "
        "decisão. Os intervalos de 95% vêm de 1.000 reamostragens desses clientes.",
    )

    def ci_text(key: str, decimals: int = 3) -> str:
        low, high = ci[key]
        return f"intervalo de 95%: {fmt_num(low, decimals)} a {fmt_num(high, decimals)}"

    components.kpis(
        [
            {
                "label": "Average precision",
                "value": fmt_num(test["average_precision"], 3),
                "note": f"acerto ao ordenar quem cancela; um palpite pela taxa média daria "
                f"{fmt_num(m['churn_rate'], 3)} · {ci_text('average_precision')}",
                "accent": True,
            },
            {
                "label": "ROC-AUC",
                "value": fmt_num(test["roc_auc"], 3),
                "note": "chance de o modelo dar mais risco a quem cancela do que a quem fica · "
                + ci_text("roc_auc"),
            },
            {
                "label": "Brier",
                "value": fmt_num(test["brier"], 3),
                "note": "erro quadrático médio das probabilidades (menor é melhor)"
                + (f"; o baseline tem {fmt_num(baseline_brier, 3)}" if baseline_brier else ""),
            },
            {
                "label": "Erro de calibração",
                "value": f"{fmt_num(test['ece'] * 100, 1)} pp",
                "note": "distância média entre a probabilidade prevista e a taxa real (ECE)",
            },
        ]
    )


def _selection(m: dict) -> None:
    cv, sel = m["cv_comparison"], m["selection"]
    components.section(
        "Escolha do modelo",
        "Por que uma regressão logística",
        "Quatro candidatos, comparados com validação cruzada (5 partes × 3 repetições) só nos "
        "dados de treino. Regra de decisão: entre os candidatos a até 1 erro-padrão do melhor, "
        "fica o mais simples — diferença menor que o ruído não justifica um modelo mais "
        "complexo, mais pesado para servir e mais difícil de explicar.",
    )
    candidates = [
        k for k in ("logistic_regression", "random_forest", "hist_gradient_boosting") if k in cv
    ]
    means = [cv[k]["cv_ap_mean"] for k in candidates]
    ses = [cv[k]["cv_ap_se"] for k in candidates]
    lo = min(mu - se for mu, se in zip(means, ses, strict=True)) - 0.004
    hi = max(mu + se for mu, se in zip(means, ses, strict=True)) + 0.004
    rows = [
        {
            "label": MODEL_NAMES[k],
            "mean": cv[k]["cv_ap_mean"],
            "se": cv[k]["cv_ap_se"],
            "chosen": k == sel["chosen"],
            "sub": COMPLEXITY[k],
        }
        for k in candidates
    ]
    chart = charts.dot_ranges(rows, (lo, hi), sel.get("floor"), lambda v: fmt_num(v, 3))
    components.card(chart)
    best = MODEL_NAMES.get(sel["best_mean"], sel["best_mean"])
    chosen = MODEL_NAMES.get(sel["chosen"], sel["chosen"])
    gap = cv[sel["best_mean"]]["cv_ap_mean"] - cv[sel["chosen"]]["cv_ap_mean"]
    components.note(
        f"<b>{esc(best)}</b> teve a maior média, mas a vantagem sobre a <b>{esc(chosen.lower())}</b> "
        f"foi de {fmt_num(gap, 4)} — menor que 1 erro-padrão ({fmt_num(cv[sel['best_mean']]['cv_ap_se'], 4)}, "
        "já corrigido para folds que se sobrepõem, Nadeau & Bengio, 2003). A linha amarela marca "
        "o limite do empate. O baseline (taxa média) fica em "
        f"{fmt_num(cv['dummy_baseline']['cv_ap_mean'], 3)}, bem à esquerda do gráfico."
        if sel["best_mean"] != sel["chosen"]
        else f"<b>{esc(chosen)}</b> teve a maior média e foi escolhida."
    )
    body = []
    for key in ("dummy_baseline", *candidates):
        r = cv[key]
        hl = ' class="cr-hl"' if key == sel["chosen"] else ""
        body.append(
            f"<tr{hl}><td>{esc(MODEL_NAMES[key])}</td>"
            f"<td>{fmt_num(r['cv_ap_mean'], 3)} ± {fmt_num(r['cv_ap_se'], 3)}</td>"
            f"<td>{fmt_num(r['oof_roc_auc'], 3)}</td><td>{fmt_num(r['oof_brier'], 3)}</td>"
            f"<td>{fmt_num(r['oof_ece'] * 100, 1)} pp</td></tr>"
        )
    with st.expander("Ver a tabela da validação cruzada", icon=":material/table_chart:"):
        ui_html(
            '<div class="cr-table-wrap"><table class="cr-table"><thead><tr><th>Candidato</th>'
            "<th>Average precision</th><th>ROC-AUC</th><th>Brier</th><th>Calibração (ECE)</th>"
            f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        )
        st.caption(
            "Average precision: média ± erro-padrão corrigido nas 15 rodadas. As demais "
            "colunas usam as previsões fora da amostra de cada candidato."
        )


def _calibration(ev: dict) -> None:
    bins = ev.get("reliability", {}).get("test", [])
    if not bins:
        return
    components.section(
        "Calibração",
        "Dá para ler a probabilidade ao pé da letra?",
        "Os clientes de teste foram agrupados por faixa de probabilidade prevista. Se o modelo é "
        "calibrado, a taxa real de cancelamento de cada grupo cai sobre a diagonal.",
    )
    pts = [(b["mean_predicted"] * 100, b["observed_rate"] * 100) for b in bins]
    hover = [
        (
            b["mean_predicted"] * 100,
            f"<b>previsto {fmt_pct(b['mean_predicted'])}</b><br>real {fmt_pct(b['observed_rate'])}"
            f"<br>{fmt_int(b['n'])} clientes",
        )
        for b in bins
    ]
    chart = charts.line_chart(
        series=[
            {"points": [(0, 0), (100, 100)], "color": MUTED, "width": 1},
            {"points": pts, "color": ACCENT},
        ],
        x_domain=(0, 100),
        y_domain=(0, 100),
        x_ticks=[(v, f"{v}%") for v in (0, 25, 50, 75, 100)],
        y_ticks=[(v, f"{v}%") for v in (0, 25, 50, 75, 100)],
        markers=[
            {"x": x, "y": y, "color": ACCENT, "label": f"previsto {x:.0f}% · real {y:.0f}%"}
            for x, y in pts
        ],
        hover=hover,
        height=280,
        x_title="probabilidade prevista (média do grupo)",
        aria="Diagrama de calibração: taxa real de cancelamento por faixa de probabilidade prevista",
    )
    left, right = st.columns([3, 2], gap="large")
    with left:
        components.card(
            charts.legend([("Modelo", ACCENT, "line"), ("Calibração perfeita", MUTED, "line")])
            + chart
        )
    with right:
        rows = "".join(
            f"<tr><td>{fmt_pct(b['low'], 0)} a {fmt_pct(b['high'], 0)}</td><td>{fmt_int(b['n'])}</td>"
            f"<td>{fmt_pct(b['mean_predicted'])}</td><td>{fmt_pct(b['observed_rate'])}</td></tr>"
            for b in bins
        )
        ui_html(
            '<div class="cr-table-wrap"><table class="cr-table" style="min-width:0"><thead><tr>'
            "<th>Faixa</th><th>Clientes</th><th>Previsto</th><th>Real</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
        components.note(
            "Na versão anterior, o modelo usava reponderação de classes e previa 38% de risco médio "
            "para uma base com 27% de cancelamento. Sem a reponderação, as probabilidades "
            "voltaram a significar o que dizem — e é isso que permite calcular o valor esperado "
            "de cada contato."
        )


def _effect_label(effect: dict) -> tuple[str, str]:
    field, category, reference = effect["field"], effect["category"], effect["reference"]
    name = FIELD_LABELS.get(field, field)
    if field == "tenure":
        return f"{name}: +12 meses", "a cada ano a mais de casa"
    if field == "SeniorCitizen":
        return "Idoso", "contra não idoso"
    if field == "InternetService" and category == "No":
        return "Sem internet", f"contra {value_label(field, reference)}"
    if category == "Yes":
        return name, "ter contra não ter"
    ref = value_label(field, reference)
    ref = ref if ref.isupper() else ref[0].lower() + ref[1:]
    return f"{name}: {value_label(field, category)}", f"contra {ref}"


def _learned(ev: dict) -> None:
    effects = ev.get("effects")
    components.section(
        "O que o modelo aprendeu",
        "Quanto cada característica muda a chance de cancelar",
        "Razão de chances de cada característica mantendo todo o resto igual: ×2 dobra a "
        "chance de cancelar; ×0,5 corta pela metade. Associação nos dados, não causa.",
    )
    if not effects:
        components.note(
            "O modelo em produção não é linear: veja a explicação por cliente na página Cliente."
        )
        return
    top = [e for e in effects if abs(e["log_odds"]) >= 0.05][:14]
    limit = max(abs(e["log_odds"]) for e in top) * 1.08
    rows = []
    for e in top:
        label, sub = _effect_label(e)
        rows.append({"label": label, "value": e["log_odds"], "sub": sub})

    def fmt_or(log_odds: float) -> str:
        return "×" + fmt_num(math.exp(log_odds), 2)

    left, right = st.columns([3, 2], gap="large")
    with left:
        components.card(charts.diverging_bars(rows, limit, fmt_or))
    with right:
        pd = ev.get("partial_dependence", {}).get("tenure")
        if pd:
            pts = list(zip(pd["grid"], [v * 100 for v in pd["average"]], strict=True))
            y_max = math.ceil(max(v for _, v in pts) / 10) * 10
            chart = charts.line_chart(
                series=[{"points": pts, "color": ACCENT, "fill": True}],
                x_domain=(0, 72),
                y_domain=(0, y_max),
                x_ticks=[
                    (0, "0"),
                    (12, "12"),
                    (24, "24"),
                    (36, "36"),
                    (48, "48"),
                    (60, "60"),
                    (72, "72"),
                ],
                y_ticks=[(v, f"{v}%") for v in range(0, y_max + 1, 10)],
                hover=[
                    (x, f"<b>{x:.0f} meses</b><br>risco médio {fmt_num(y, 1)}%") for x, y in pts
                ],
                height=230,
                x_title="tempo de casa (meses)",
                aria="Risco médio previsto por tempo de casa",
            )
            components.card(chart, title="Risco médio conforme o tempo de casa")
        gender = next((e for e in effects if e["field"] == "gender"), None)
        if gender:
            components.note(
                f"<b>Gênero</b> quase não pesa (×{fmt_num(gender['odds_ratio'], 2)}): o modelo "
                "trata clientes de gêneros diferentes, com o mesmo perfil, praticamente igual."
            )


def _subgroups(ev: dict) -> None:
    groups = ev.get("subgroup_metrics", {})
    if not groups:
        return
    components.section(
        "Onde o modelo erra mais",
        "Desempenho por grupo de clientes",
        f"Métricas no corte em produção ({fmt_pct(ev['optimal_threshold'], 0)}). Uma média boa "
        "pode esconder um grupo em que o modelo vai mal.",
    )
    rows = []
    for key, label in SUBGROUPS:
        g = groups.get(key)
        if not g:
            continue
        auc = fmt_num(g["roc_auc"], 3) if "roc_auc" in g else "—"
        rows.append(
            [
                esc(label),
                fmt_int(g["n"]),
                fmt_pct(g["churn_rate"]),
                fmt_pct(g["mean_predicted"]),
                fmt_pct(g["contact_rate"]),
                fmt_pct(g["recall"]),
                fmt_pct(g["precision"]),
                auc,
            ]
        )
    ui_html(
        components.table(
            [
                "Grupo",
                "Clientes",
                "Churn real",
                "Previsto",
                "Contatados",
                "Encontrados",
                "Precisão",
                "ROC-AUC",
            ],
            rows,
        )
    )
    two_year = groups.get("contract_two_year")
    if two_year:
        components.note(
            f"<b>Contrato bienal:</b> só {fmt_pct(two_year['churn_rate'])} desses clientes cancelam, "
            "e o modelo praticamente nunca recomenda contato — os poucos cancelamentos desse grupo "
            "passam despercebidos. Para esse perfil, uma regra de negócio (por exemplo, contato "
            "perto do fim do contrato) funciona melhor que o modelo.",
            kind="warn",
        )


def _use(m: dict) -> None:
    components.section("Uso", "Para que serve — e para que não serve")
    ok = [
        "Priorizar quem a equipe de retenção contata primeiro.",
        "Estimar o retorno de uma campanha antes de rodá-la.",
        "Mostrar, cliente a cliente, o que puxa o risco para cima.",
        "Apontar hipóteses de ação para testar com grupo de controle.",
    ]
    no = [
        "Negar serviço, mudar preço ou tratar pior um cliente.",
        "Provar que uma ação causa retenção (o modelo mede associação).",
        "Prever clientes de outra empresa ou outro país sem retreinar.",
        "Substituir o julgamento de quem conhece o cliente.",
    ]
    ui_html(
        '<div class="cr-decisions">'
        '<div class="cr-card" style="border-top:2px solid #34D399"><p class="cr-card-title" style="color:#34D399">Serve para</p>'
        + "".join(f'<p class="cr-text" style="margin:0 0 6px 0">✓ {esc(t)}</p>' for t in ok)
        + "</div>"
        '<div class="cr-card" style="border-top:2px solid #F87171"><p class="cr-card-title" style="color:#F87171">Não serve para</p>'
        + "".join(f'<p class="cr-text" style="margin:0 0 6px 0">✕ {esc(t)}</p>' for t in no)
        + "</div></div>"
    )
    components.note(
        "<b>Limitações:</b> dataset público IBM Telco Customer Churn — "
        f"{fmt_int(m.get('n_samples', 7032))} clientes de uma operadora dos EUA, um retrato de um "
        "único momento; os custos da campanha são hipóteses; e o perfil de uma carteira nova "
        "pode ser diferente do treino (a página Carteira mede isso com o PSI)."
    )


def _technical(m: dict, ev: dict) -> None:
    with st.expander("Detalhes técnicos e reprodutibilidade", icon=":material/terminal:"):
        excluded = ", ".join(m.get("excluded_features", [])) or "nenhuma"
        rows = [
            ("Dados", f"{fmt_int(m['n_samples'])} clientes (11 sem total gasto descartados)"),
            (
                "Divisão",
                f"{fmt_int(m['n_train'])} treino / {fmt_int(m['n_test'])} teste, estratificada, semente 42",
            ),
            ("Validação", "RepeatedStratifiedKFold 5 × 3, erro-padrão corrigido (Nadeau & Bengio)"),
            ("Seleção", "regra de 1 erro-padrão sobre a average precision"),
            (
                "Corte de decisão",
                f"{fmt_pct(ev['optimal_threshold'], 0)}, escolhido nas previsões fora da amostra do treino",
            ),
            ("Variáveis fora do modelo", excluded),
            ("Rastreamento", "MLflow (cada candidato e o modelo final)"),
            (
                "Versões",
                f"Python {m.get('python', '—')} · scikit-learn {m.get('scikit_learn', '—')}",
            ),
        ]
        ui_html(
            '<div class="cr-table-wrap"><table class="cr-table" style="min-width:0"><tbody>'
            + "".join(
                f'<tr><td>{esc(k)}</td><td style="text-align:left;white-space:normal">{esc(v)}</td></tr>'
                for k, v in rows
            )
            + "</tbody></table></div>"
        )
        st.caption("Para reproduzir o treino e os testes:")
        st.code("python -m src.train\npytest", language="bash")


def render() -> None:
    m, ev = metrics(), evaluation()
    components.page_header(
        "Model card",
        "Como o modelo foi construído",
        "Tudo o que dá para saber sobre o modelo em produção: como ele foi escolhido, quão "
        "confiáveis são as probabilidades, o que ele aprendeu, onde erra mais e para que ele "
        "não serve.",
    )
    if not m or not ev:
        components.note("Rode <code>python -m src.train</code> para gerar o model card.", "warn")
        return
    _meta(m)
    _test_results(m)
    _selection(m)
    _calibration(ev)
    _learned(ev)
    _subgroups(ev)
    _use(m)
    _technical(m, ev)
