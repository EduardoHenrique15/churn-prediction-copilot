"""Página "Carteira": previsão em lote, fila de contato e monitor de drift."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score

from src.business import (
    PSI_MIN_ROWS,
    PSI_SHIFT,
    PSI_STABLE,
    drift_report,
    expected_contact_value,
    fmt_brl,
    fmt_int,
    fmt_num,
    fmt_pct,
)
from src.client import InvalidCustomerError
from src.labels import FIELD_LABELS, VALUE_LABELS
from src.ui import charts, components
from src.ui.data import EXAMPLE_FILES, evaluation, example_batch, get_client, reference_profile
from src.ui.html import esc, md_text, ui_html
from src.ui.theme import DANGER, OK, WARN
from src.utils import RAW_INPUT_COLUMNS
from src.validation import prepare_batch

MAX_ROWS = 20_000
SOURCES = {
    "tipica": "Exemplo: base típica",
    "novos": "Exemplo: clientes novos",
    "upload": "Enviar CSV",
}


def _read_upload(file) -> pd.DataFrame:
    raw = file.getvalue()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    header = text.splitlines()[0]
    # Planilhas em português costumam exportar com ";" e vírgula decimal.
    if header.count(";") > header.count(","):
        return pd.read_csv(io.StringIO(text), sep=";", decimal=",")
    return pd.read_csv(io.StringIO(text))


def _source() -> tuple[pd.DataFrame | None, str]:
    choice = st.segmented_control(
        "Base de clientes",
        list(SOURCES),
        format_func=SOURCES.get,
        default="tipica",
        required=True,
        key="b_source",
    )
    if choice in EXAMPLE_FILES:
        _, title, description = EXAMPLE_FILES[choice]
        st.caption(md_text(f"{title}: {description}. Inclui o desfecho real (coluna Churn)."))
        return example_batch(choice), title

    st.caption(
        "Use as colunas do dataset IBM Telco Customer Churn, com os valores originais em inglês. "
        "`customerID` e `Churn` são opcionais. Até 20 mil linhas."
    )
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    file = col1.file_uploader("Arquivo CSV", type="csv", key="b_file", label_visibility="collapsed")
    col2.download_button(
        "Baixar um modelo",
        example_batch("tipica").head(20).to_csv(index=False).encode("utf-8"),
        "modelo_carteira.csv",
        "text/csv",
        icon=":material/download:",
        width="stretch",
    )
    if file is None:
        return None, ""
    try:
        return _read_upload(file), file.name
    except Exception:
        components.note(
            "Não consegui ler esse arquivo como CSV. Confira o formato e tente de novo.", "warn"
        )
        return None, ""


def _predict(records: list[dict]) -> tuple[list[dict], str, float]:
    key = hashlib.sha1(pd.DataFrame(records).to_csv(index=False).encode()).hexdigest()
    cache = st.session_state.setdefault("batch_cache", {})
    if key not in cache:
        answer = get_client().predict(records)
        cache.clear()
        cache[key] = (answer.data, answer.source, answer.ms)
    return cache[key]


def _validation_report(n_rows: int, n_valid: int, problems: pd.DataFrame, notes: list[str]) -> None:
    kind = "ok" if problems.empty else "warn"
    text = f"<b>{fmt_int(n_rows)}</b> linhas lidas · <b>{fmt_int(n_valid)}</b> prontas para prever"
    if not problems.empty:
        text += f" · <b>{fmt_int(len(problems))}</b> com problema (ignoradas)"
    for extra in notes:
        text += f"<br>{esc(extra)}"
    components.note(text, kind)
    if not problems.empty:
        with st.expander(f"Ver as {len(problems)} linhas com problema", icon=":material/error:"):
            st.dataframe(
                problems,
                hide_index=True,
                column_config={
                    "linha": st.column_config.NumberColumn("Linha", format="%d", width="small"),
                    "problemas": st.column_config.TextColumn("O que corrigir", width="large"),
                },
            )
            st.download_button(
                "Baixar a lista de problemas",
                problems.to_csv(index=False).encode("utf-8"),
                "problemas.csv",
                "text/csv",
                icon=":material/download:",
            )


def _score(
    clean: pd.DataFrame, records: list[dict], predictions: list[dict], costs: dict
) -> pd.DataFrame:
    scored = clean.copy()
    probs = np.array([p["churn_probability"] for p in predictions])
    scored["probabilidade_churn"] = probs
    scored["risco"] = [p["risk_level"] for p in predictions]
    scored["contatar"] = ["Sim" if p["churn_prediction"] else "Não" for p in predictions]
    scored["valor_esperado_contato"] = [
        round(
            expected_contact_value(p, costs["ltv"], costs["offer_cost"], costs["success_rate"]), 2
        )
        for p in probs
    ]
    order = scored["valor_esperado_contato"].rank(ascending=False, method="first").astype(int)
    scored["prioridade"] = order
    scored["_monthly"] = [r["MonthlyCharges"] for r in records]
    return scored


def _summary(scored: pd.DataFrame) -> None:
    probs = scored["probabilidade_churn"].to_numpy()
    contact = scored["contatar"] == "Sim"
    campaign = scored.loc[contact, "valor_esperado_contato"].sum()
    components.kpis(
        [
            {
                "label": "A contatar",
                "value": fmt_int(contact.sum()),
                "note": f"{fmt_pct(contact.mean())} da carteira, em que contatar compensa",
                "accent": True,
            },
            {
                "label": "Cancelamentos esperados",
                "value": fmt_int(probs.sum()),
                "note": f"se nada for feito — {fmt_pct(probs.mean())} da carteira",
            },
            {
                "label": "Receita mensal em risco",
                "value": fmt_brl(float((probs * scored["_monthly"]).sum())),
                "note": "soma de risco × mensalidade de cada cliente",
            },
            {
                "label": "Valor esperado da campanha",
                "value": fmt_brl(float(campaign)),
                "note": "contatando só quem está acima do corte, com as hipóteses de referência "
                "de custo (ajustáveis na página Estratégia)",
            },
        ]
    )


def _distribution(scored: pd.DataFrame) -> None:
    ev = evaluation()
    counts = scored["risco"].value_counts()
    components.section("Distribuição", "Como o risco se espalha na carteira")
    bar = charts.stacked_bar(
        [
            {"label": "Risco baixo", "n": int(counts.get("Baixo", 0)), "color": OK},
            {"label": "Risco médio", "n": int(counts.get("Médio", 0)), "color": WARN},
            {"label": "Risco alto", "n": int(counts.get("Alto", 0)), "color": DANGER},
        ]
    )
    scale = charts.risk_scale(
        scored["probabilidade_churn"].to_numpy(),
        ev.get("risk_level_cuts", {"baixo_max": 0.5, "medio_max": 0.5}),
        threshold=ev.get("optimal_threshold"),
        threshold_label=f"corte de contato ({fmt_pct(ev.get('optimal_threshold', 0.5), 0)})",
        population="clientes da carteira",
    )
    components.card(bar + '<div style="height:10px"></div>' + scale)


def _truth(scored: pd.DataFrame) -> None:
    if "Churn" not in scored.columns:
        return
    truth = scored["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    if truth.isna().any() or truth.nunique() < 2:
        return
    y = truth.to_numpy()
    p = scored["probabilidade_churn"].to_numpy()
    contact = (scored["contatar"] == "Sim").to_numpy()
    tp = int((contact & (y == 1)).sum())
    components.section(
        "Conferência",
        "O modelo contra o que aconteceu de verdade",
        "O arquivo traz a coluna Churn com o desfecho real, então dá para medir o acerto nesta "
        "carteira. Nos exemplos, são clientes do conjunto de teste — o modelo não os viu.",
    )
    components.kpis(
        [
            {
                "label": "Churn real × previsto",
                "value": f"{fmt_pct(y.mean())} × {fmt_pct(p.mean())}",
                "note": "taxa real de cancelamento contra a média das probabilidades",
                "accent": True,
            },
            {
                "label": "Encontrados",
                "value": fmt_pct(tp / max(y.sum(), 1), 0),
                "note": f"{fmt_int(tp)} de {fmt_int(y.sum())} cancelamentos estavam na lista de contato",
            },
            {
                "label": "Precisão da lista",
                "value": fmt_pct(tp / max(contact.sum(), 1), 0),
                "note": "dos contatados, quantos iam mesmo cancelar",
            },
            {
                "label": "Average precision",
                "value": fmt_num(average_precision_score(y, p), 3),
                "note": f"contra {fmt_num(y.mean(), 3)} de um palpite pela taxa média",
            },
        ]
    )


def _queue(scored: pd.DataFrame, name: str) -> None:
    components.section(
        "Fila de contato",
        "Quem contatar primeiro",
        "Ordenada pelo valor esperado do contato. Clique no cabeçalho de uma coluna para "
        "reordenar.",
    )
    view = scored.sort_values("prioridade").copy()
    view["risco_pct"] = view["probabilidade_churn"] * 100
    view["Contrato"] = view["Contract"].map(VALUE_LABELS["Contract"])
    view["Internet"] = view["InternetService"].map(VALUE_LABELS["InternetService"])
    view["Pagamento"] = view["PaymentMethod"].map(VALUE_LABELS["PaymentMethod"])
    only_contact = st.toggle("Mostrar só quem deve ser contatado", value=True, key="b_only")
    columns = ["prioridade", "risco_pct", "risco", "valor_esperado_contato"]
    if not only_contact:
        columns.insert(3, "contatar")
    if "customerID" in view.columns:
        columns.insert(1, "customerID")
    columns += ["tenure", "Contrato", "Internet", "Pagamento", "MonthlyCharges"]
    if only_contact:
        view = view[view["contatar"] == "Sim"]
    st.dataframe(
        view[columns],
        hide_index=True,
        height=min(38 + 35 * len(view), 460),
        column_config={
            "prioridade": st.column_config.NumberColumn("#", format="%d", width=48),
            "customerID": st.column_config.TextColumn("Cliente", width=110),
            "risco_pct": st.column_config.ProgressColumn(
                "Risco", format="%.0f%%", min_value=0.0, max_value=100.0, width=120
            ),
            "risco": st.column_config.TextColumn("Faixa", width=70),
            "contatar": st.column_config.TextColumn("Contatar", width=80),
            "valor_esperado_contato": st.column_config.NumberColumn(
                "Valor esperado", format="R$ %.0f", width=110
            ),
            "tenure": st.column_config.NumberColumn("Meses", format="%d", width=70),
            "Contrato": st.column_config.TextColumn("Contrato", width=90),
            "Internet": st.column_config.TextColumn("Internet", width=100),
            "MonthlyCharges": st.column_config.NumberColumn(
                "Mensalidade", format="R$ %.0f", width=100
            ),
        },
    )
    export = scored.drop(columns=["_monthly"]).sort_values("prioridade")
    c1, c2 = st.columns(2)
    stem = name.rsplit(".", 1)[0].replace(" ", "_").lower() or "carteira"
    c1.download_button(
        "Baixar o resultado completo (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        f"{stem}_com_risco.csv",
        "text/csv",
        icon=":material/download:",
        width="stretch",
    )
    c2.download_button(
        "Baixar só a fila de contato (CSV)",
        export[export["contatar"] == "Sim"].to_csv(index=False).encode("utf-8"),
        f"{stem}_fila_de_contato.csv",
        "text/csv",
        icon=":material/call:",
        width="stretch",
    )


def _drift(records: list[dict]) -> None:
    components.section(
        "Monitor de mudanças",
        "Esta carteira se parece com os clientes de treino?",
        "Quando o perfil muda muito, as previsões merecem mais cautela. A comparação usa o PSI "
        "(Population Stability Index) de cada variável: abaixo de 0,10 é estável, de 0,10 a 0,25 "
        "pede atenção e acima de 0,25 indica mudança forte.",
    )
    if len(records) < PSI_MIN_ROWS:
        components.note(
            f"São necessários pelo menos {PSI_MIN_ROWS} clientes para uma comparação confiável "
            f"(este arquivo tem {len(records)})."
        )
        return
    profile = reference_profile()
    if not profile:
        components.note(
            "Perfil de referência ausente: rode <code>python -m src.train</code>.", "warn"
        )
        return
    report = drift_report(pd.DataFrame(records), profile)
    rows = [
        {
            "label": FIELD_LABELS.get(r["variavel"], r["variavel"]),
            "psi": r["psi"],
            "status": r["status"],
        }
        for r in report[:8]
    ]
    strong = [r for r in report if r["status"] == "Mudança forte"]
    components.card(charts.psi_bars(rows, PSI_STABLE, PSI_SHIFT))
    if strong:
        names = ", ".join(FIELD_LABELS.get(r["variavel"], r["variavel"]).lower() for r in strong)
        components.note(
            f"<b>Mudança forte em {len(strong)} variável(is):</b> {esc(names)}. Esta carteira "
            "é diferente dos clientes que o modelo conheceu — confira a calibração com dados "
            "reais antes de confiar nas probabilidades e considere retreinar.",
            kind="warn",
        )
    else:
        components.note(
            "Perfil parecido com o de treino: nenhuma variável com mudança forte.", "ok"
        )


def render() -> None:
    components.page_header(
        "Análise de carteira",
        "Quem contatar primeiro",
        "Envie a lista de clientes (CSV) ou use um exemplo. O Radar valida linha a linha, calcula "
        "o risco de todos, monta a fila de contato e compara o perfil da carteira com os "
        "clientes de treino.",
    )
    df, name = _source()
    if df is None:
        return
    if len(df) > MAX_ROWS:
        components.note(
            f"O arquivo tem {fmt_int(len(df))} linhas; o limite é {fmt_int(MAX_ROWS)}.", "warn"
        )
        return
    try:
        clean, records, problems, notes = prepare_batch(df)
    except ValueError as exc:
        missing = str(exc)
        components.note(
            f"<b>{esc(missing)}.</b> O arquivo precisa das colunas: "
            + ", ".join(f"<code>{c}</code>" for c in RAW_INPUT_COLUMNS),
            "warn",
        )
        return
    _validation_report(len(df), len(records), problems, notes)
    if not records:
        return

    try:
        with st.spinner(f"Calculando o risco de {fmt_int(len(records))} clientes…"):
            predictions, source, ms = _predict(records)
    except InvalidCustomerError:
        components.note("A validação da API recusou este lote.", "warn")
        return

    costs = evaluation().get(
        "cost_assumptions", {"ltv": 1000.0, "offer_cost": 100.0, "success_rate": 0.3}
    )
    scored = _score(clean, records, predictions, costs)
    ui_html(
        f'<p class="cr-sub" style="margin:4px 0 12px 0">{fmt_int(len(records))} clientes · '
        f"{esc(components.source_label(source, ms))}</p>"
    )
    _summary(scored)
    _distribution(scored)
    _truth(scored)
    _queue(scored, name)
    _drift(records)
