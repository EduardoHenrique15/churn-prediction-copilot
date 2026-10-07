"""Página "Cliente": análise individual — risco, decisão, explicação e cenários."""

from __future__ import annotations

import json
import random
from contextlib import suppress

import streamlit as st

from src.business import (
    expected_contact_value,
    fmt_brl,
    fmt_int,
    fmt_pct,
    whatif_scenarios,
)
from src.client import ExplanationUnavailableError, InvalidCustomerError
from src.labels import VALUE_LABELS, field_contributions
from src.ui import charts, components
from src.ui.data import evaluation, example_batch, get_client, metrics, test_predictions
from src.ui.html import esc, md_text, ui_html
from src.ui.theme import RISK_STYLE
from src.utils import (
    CATEGORY_OPTIONS,
    INTERNET_SERVICES,
    consistency_errors,
    expected_total_charges,
)

SERVICE_LABELS = {
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
}

# Perfis de exemplo, já no formato dos controles do formulário.
PROFILES = {
    "Alto risco": {
        "gender": "Female", "senior": False, "partner": False, "dependents": False,
        "tenure": 2, "contract": "Month-to-month", "payment": "Electronic check", "paperless": True,
        "phone": True, "multiple": False, "internet": "Fiber optic", "services": ["StreamingTV"],
        "monthly": 89.9,
    },
    "Risco médio": {
        "gender": "Female", "senior": False, "partner": True, "dependents": False,
        "tenure": 14, "contract": "Month-to-month", "payment": "Electronic check", "paperless": True,
        "phone": True, "multiple": False, "internet": "DSL", "services": [],
        "monthly": 54.2,
    },
    "Baixo risco": {
        "gender": "Male", "senior": False, "partner": True, "dependents": False,
        "tenure": 34, "contract": "One year", "payment": "Credit card (automatic)", "paperless": True,
        "phone": True, "multiple": False, "internet": "DSL", "services": ["OnlineSecurity"],
        "monthly": 59.9,
    },
}  # fmt: skip
REAL_CUSTOMER = "Cliente real"
FLAGS = {"senior": "Idoso (65+)", "partner": "Tem cônjuge", "dependents": "Tem dependentes"}
RISK_TAG = {"Baixo": "ok", "Médio": "warn", "Alto": "danger"}
PERSIST = {"persist_state": "session"}
DEFAULT_PROFILE = "Alto risco"
MAX_CACHE = 40


# ---------------------------------------------------------------------------
# Estado do formulário
# ---------------------------------------------------------------------------
def _apply(values: dict) -> None:
    for name, value in values.items():
        if name not in FLAGS:
            st.session_state[f"f_{name}"] = value
    st.session_state["f_flags"] = [flag for flag in FLAGS if values[flag]]
    st.session_state["f_total_auto"] = True
    st.session_state["f_total"] = expected_total_charges(values["tenure"], values["monthly"])


def _from_payload(payload: dict) -> dict:
    """Cliente cru (dataset) -> valores dos controles do formulário."""
    has_internet = payload["InternetService"] != "No"
    return {
        "gender": payload["gender"],
        "senior": int(payload["SeniorCitizen"]) == 1,
        "partner": payload["Partner"] == "Yes",
        "dependents": payload["Dependents"] == "Yes",
        "tenure": int(payload["tenure"]),
        "contract": payload["Contract"],
        "payment": payload["PaymentMethod"],
        "paperless": payload["PaperlessBilling"] == "Yes",
        "phone": payload["PhoneService"] == "Yes",
        "multiple": payload["MultipleLines"] == "Yes",
        "internet": payload["InternetService"],
        "services": [s for s in INTERNET_SERVICES if has_internet and payload[s] == "Yes"],
        "monthly": float(payload["MonthlyCharges"]),
    }


def _load_profile() -> None:
    choice = st.session_state.get("f_profile")
    st.session_state["f_real"] = None
    if choice == REAL_CUSTOMER:
        df = example_batch("tipica")
        row = df.iloc[random.randrange(len(df))].to_dict()
        values = _from_payload(row)
        _apply(values)
        st.session_state["f_total_auto"] = False
        st.session_state["f_total"] = float(row["TotalCharges"])
        st.session_state["f_real"] = {
            "id": row["customerID"],
            "churn": row["Churn"],
            "values": values,
        }
    elif choice in PROFILES:
        _apply(PROFILES[choice])
    st.session_state["f_profile"] = None


def _init_state() -> None:
    if "f_tenure" not in st.session_state:
        _apply(PROFILES[DEFAULT_PROFILE])
        st.session_state["f_real"] = None


def _payload() -> dict:
    s = st.session_state
    has_internet = s.f_internet != "No"
    services = set(s.f_services or [])
    flags = set(s.f_flags or [])
    total = expected_total_charges(s.f_tenure, s.f_monthly) if s.f_total_auto else float(s.f_total)
    payload = {
        "gender": s.f_gender,
        "SeniorCitizen": int("senior" in flags),
        "Partner": "Yes" if "partner" in flags else "No",
        "Dependents": "Yes" if "dependents" in flags else "No",
        "tenure": int(s.f_tenure),
        "PhoneService": "Yes" if s.f_phone else "No",
        "MultipleLines": ("Yes" if s.f_multiple else "No") if s.f_phone else "No phone service",
        "InternetService": s.f_internet,
    }
    for field in INTERNET_SERVICES:
        payload[field] = (
            ("Yes" if field in services else "No") if has_internet else "No internet service"
        )
    payload.update(
        {
            "Contract": s.f_contract,
            "PaperlessBilling": "Yes" if s.f_paperless else "No",
            "PaymentMethod": s.f_payment,
            "MonthlyCharges": round(float(s.f_monthly), 2),
            "TotalCharges": round(total, 2),
        }
    )
    return payload


def _label(field: str):
    return lambda value: VALUE_LABELS.get(field, {}).get(value, value)


def _on_internet_change() -> None:
    if st.session_state.f_internet == "No":
        st.session_state.f_services = []


def _on_phone_change() -> None:
    if not st.session_state.f_phone:
        st.session_state.f_multiple = False


def _on_total_auto_change() -> None:
    """Ao desligar o cálculo automático, o campo começa no valor esperado."""
    if not st.session_state.f_total_auto:
        s = st.session_state
        s.f_total = expected_total_charges(s.f_tenure, s.f_monthly)


# ---------------------------------------------------------------------------
# Formulário
# ---------------------------------------------------------------------------
def _form() -> None:
    st.pills(
        "Carregar um exemplo",
        [*PROFILES, REAL_CUSTOMER],
        key="f_profile",
        on_change=_load_profile,
        wrap=True,
        help='Os três primeiros são perfis montados. "Cliente real" sorteia um cliente do '
        "conjunto de teste e mostra o que aconteceu com ele de verdade.",
    )
    with st.container(key="panel"):
        ui_html('<p class="cr-group">Relacionamento</p>')
        st.slider("Tempo de casa (meses)", 0, 72, key="f_tenure", **PERSIST)
        st.segmented_control(
            "Contrato",
            list(CATEGORY_OPTIONS["Contract"]),
            key="f_contract",
            format_func={"Month-to-month": "Mensal", "One year": "Anual", "Two year": "Bienal"}.get,
            required=True,
            width="stretch",
            **PERSIST,
        )
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        c1.selectbox(
            "Forma de pagamento",
            list(CATEGORY_OPTIONS["PaymentMethod"]),
            key="f_payment",
            format_func=_label("PaymentMethod"),
            **PERSIST,
        )
        c2.toggle("Fatura digital", key="f_paperless", **PERSIST)

        ui_html('<p class="cr-group">Serviços</p>')
        st.segmented_control(
            "Internet",
            list(CATEGORY_OPTIONS["InternetService"]),
            key="f_internet",
            format_func={"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sem internet"}.get,
            required=True,
            width="stretch",
            on_change=_on_internet_change,
            **PERSIST,
        )
        has_internet = st.session_state.f_internet != "No"
        st.pills(
            "Serviços adicionais",
            list(SERVICE_LABELS),
            selection_mode="multi",
            key="f_services",
            format_func=SERVICE_LABELS.get,
            disabled=not has_internet,
            help="Só existem para quem tem internet.",
            **PERSIST,
        )
        c1, c2 = st.columns(2)
        c1.toggle("Telefone", key="f_phone", on_change=_on_phone_change, **PERSIST)
        c2.toggle(
            "Múltiplas linhas", key="f_multiple", disabled=not st.session_state.f_phone, **PERSIST
        )

        ui_html('<p class="cr-group">Perfil</p>')
        st.segmented_control(
            "Gênero",
            list(CATEGORY_OPTIONS["gender"]),
            key="f_gender",
            format_func=_label("gender"),
            required=True,
            width="stretch",
            **PERSIST,
        )
        st.pills(
            "Características",
            list(FLAGS),
            selection_mode="multi",
            key="f_flags",
            format_func=FLAGS.get,
            **PERSIST,
        )

        ui_html('<p class="cr-group">Cobrança</p>')
        st.number_input(
            "Mensalidade (R$)",
            min_value=18.0,
            max_value=120.0,
            step=0.5,
            format="%.2f",
            key="f_monthly",
            **PERSIST,
        )
        st.toggle(
            "Calcular o total gasto (tempo de casa × mensalidade)",
            key="f_total_auto",
            help="No dataset, o total gasto fica muito perto dessa conta.",
            on_change=_on_total_auto_change,
            **PERSIST,
        )
        if not st.session_state.f_total_auto:
            st.number_input(
                "Total gasto (R$)",
                min_value=0.0,
                max_value=10_000.0,
                step=10.0,
                format="%.2f",
                key="f_total",
                **PERSIST,
            )
        else:
            total = expected_total_charges(st.session_state.f_tenure, st.session_state.f_monthly)
            st.caption(md_text(f"Total gasto: {fmt_brl(total, cents=True)}"))
        st.caption(
            "Mensalidade e total gasto são validados, mas não entram no modelo: o preço já está "
            "nos serviços contratados (veja a página Modelo)."
        )


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------
def _analyze(payload: dict) -> dict:
    """Previsão do cliente + cenários (uma chamada) e explicação (outra)."""
    key = json.dumps(payload, sort_keys=True)
    cache = st.session_state.setdefault("analysis_cache", {})
    if key in cache:
        return cache[key]

    client = get_client()
    scenarios = whatif_scenarios(payload)
    answer = client.predict([payload, *[s for _, s in scenarios]])
    result = {
        "prediction": answer.data[0],
        "scenarios": [
            {"label": label, "prediction": pred}
            for (label, _), pred in zip(scenarios, answer.data[1:], strict=True)
        ],
        "source": answer.source,
        "ms": answer.ms,
        "explanation": None,
    }
    with suppress(ExplanationUnavailableError):
        result["explanation"] = client.explain(payload).data
    cache[key] = result
    while len(cache) > MAX_CACHE:
        cache.pop(next(iter(cache)))
    return result


def _verdict(payload: dict, result: dict) -> None:
    pred = result["prediction"]
    p = pred["churn_probability"]
    risk = pred["risk_level"]
    style = RISK_STYLE[risk]
    costs = evaluation().get(
        "cost_assumptions", {"ltv": 1000.0, "offer_cost": 100.0, "success_rate": 0.3}
    )
    value = expected_contact_value(p, costs["ltv"], costs["offer_cost"], costs["success_rate"])
    base_rate = metrics().get("churn_rate", 0.2658)
    test = test_predictions()
    percentile = float((test < p).mean()) if len(test) else None

    if pred["churn_prediction"] and value <= 0:
        # Entre o corte escolhido nos dados (ex.: 32%) e o teórico (33%), o
        # valor esperado fica praticamente em zero: melhor dizer isso do que
        # recomendar contato com um valor negativo ao lado.
        reco_icon = components.icon("call", 17)
        reco_cls = "go"
        title = "Contatar, no limite"
        text = (
            f"Este cliente está na fronteira do corte: o valor esperado do contato é praticamente "
            f"zero (<b>{fmt_brl(value)}</b>). O corte foi escolhido pelo maior retorno nos dados "
            "de treino."
        )
    elif pred["churn_prediction"]:
        reco_icon = components.icon("call", 17)
        reco_cls = "go"
        title = "Contatar com uma oferta de retenção"
        text = (
            f"Valor esperado da ação: <b>{fmt_brl(value)}</b> — {fmt_pct(p, 0)} de risco × "
            f"{fmt_pct(costs['success_rate'], 0)} de sucesso × {fmt_brl(costs['ltv'])} de valor do "
            f"cliente, menos {fmt_brl(costs['offer_cost'])} da oferta."
        )
    else:
        reco_icon = components.icon("hold", 17)
        reco_cls = "hold"
        title = "Não priorizar agora"
        text = (
            f"A oferta custaria mais do que o retorno esperado: <b>{fmt_brl(value)}</b> por "
            "contato. O risco é baixo demais para compensar."
        )

    meta = [f"Média da base: <b>{fmt_pct(base_rate)}</b>"]
    if percentile is not None:
        meta.append(f"Mais arriscado que <b>{fmt_pct(percentile, 0)}</b> dos clientes")
    meta.append(
        f"Receita mensal em risco: <b>{fmt_brl(p * payload['MonthlyCharges'], cents=True)}</b>"
    )
    meta.append(f"Previsão: <b>{esc(components.source_label(result['source'], result['ms']))}</b>")

    whole, _, decimal = f"{p * 100:.1f}".partition(".")
    ui_html(
        '<div class="cr-verdict">'
        '<div class="cr-verdict-top"><div>'
        f'<div class="cr-prob">{whole}<small>,{decimal}%</small></div>'
        '<div class="cr-prob-label">de chance de cancelar</div></div>'
        f'<span class="cr-risk cr-risk--{style["css"]}">{style["icon"]} Risco {risk.lower()}</span>'
        "</div>"
        '<div class="cr-reco">'
        f'<div class="cr-reco-icon cr-reco-icon--{reco_cls}">{reco_icon}</div>'
        f'<div><div class="cr-reco-title">{title}</div><div class="cr-reco-text">{text}</div></div>'
        "</div>"
        f'<div class="cr-meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>'
        "</div>"
    )

    real = st.session_state.get("f_real")
    if real and real["values"] == _from_payload(payload):
        outcome = "cancelou" if real["churn"] == "Yes" else "não cancelou"
        hit = (real["churn"] == "Yes") == pred["churn_prediction"]
        components.note(
            f"Cliente real <code>{esc(real['id'])}</code> do conjunto de teste — o modelo nunca o "
            f"viu no treino. Na vida real, ele <b>{outcome}</b>. "
            + (
                "A recomendação do modelo bate com o desfecho."
                if hit
                else "Aqui o modelo errou: nenhuma previsão acerta todos os casos."
            ),
            kind="ok" if hit else "warn",
        )


def _explanation(payload: dict, result: dict) -> None:
    explanation = result["explanation"]
    components.section(
        "Por quê",
        "O que mais pesou na previsão",
        "Cada barra mostra quanto uma característica deste cliente empurrou o risco para cima "
        "ou para baixo, a partir de um cliente com características médias.",
    )
    if not explanation:
        components.note("A explicação não está disponível nesta instância.", kind="warn")
        return
    rows = field_contributions(explanation["contributions"], payload)
    top, rest = rows[:7], rows[7:]
    link = explanation.get("link", "logit")

    def to_p(value: float) -> float:
        return charts.logit_to_p(value) if link == "logit" else min(max(value, 0.0), 1.0)

    cumulative = explanation["base_value"]
    steps = []
    for row in top:
        start = cumulative
        cumulative += row["contribution"]
        steps.append(
            {
                "label": row["label"],
                "value": row["value"],
                "p_start": to_p(start),
                "p_end": to_p(cumulative),
            }
        )
    rest_total = sum(r["contribution"] for r in rest)
    if rest and abs(rest_total) > 1e-9:
        start = cumulative
        cumulative += rest_total
        steps.append(
            {
                "label": f"Outras {len(rest)} características",
                "value": ", ".join(r["label"].lower() for r in rest[:3])
                + ("…" if len(rest) > 3 else ""),
                "p_start": to_p(start),
                "p_end": to_p(cumulative),
            }
        )
    components.card(charts.waterfall(to_p(explanation["base_value"]), steps, to_p(cumulative)))


def _scenarios(payload: dict, result: dict) -> None:
    components.section(
        "E se…?",
        "O que mudaria a previsão",
        "Ofertas que a empresa poderia fazer a este cliente e a previsão do modelo para cada uma.",
    )
    p = result["prediction"]["churn_probability"]
    decided = result["prediction"]["churn_prediction"]
    rows = []
    for sc in result["scenarios"]:
        new_p = sc["prediction"]["churn_probability"]
        if abs(new_p - p) < 0.005:
            continue
        tag, kind = "", "accent"
        if decided and not sc["prediction"]["churn_prediction"]:
            tag, kind = "sai da lista de contato", "ok"
        elif sc["prediction"]["risk_level"] != result["prediction"]["risk_level"]:
            tag = f"risco {sc['prediction']['risk_level'].lower()}"
            kind = RISK_TAG[sc["prediction"]["risk_level"]]
        rows.append({"label": sc["label"], "p": new_p, "tag": tag, "tag_kind": kind})
    if not rows:
        components.note(
            "Nenhuma das ofertas simuladas muda a previsão deste cliente de forma relevante."
        )
        return
    rows.sort(key=lambda r: r["p"])
    components.card(charts.whatif(p, rows))
    components.note(
        "O modelo aprende <b>associações</b> dos dados, não causas: quem já tem contrato longo "
        "costuma ser um cliente diferente em muitos aspectos. Use os cenários para escolher o que "
        "testar, não como garantia de resultado."
    )


def _api_details(payload: dict, result: dict) -> None:
    with st.expander("Ver a chamada à API (JSON e cURL)", icon=":material/data_object:"):
        st.caption("Corpo da requisição POST /predict — os valores são os do dataset original:")
        st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
        st.caption("Resposta:")
        st.code(json.dumps(result["prediction"], indent=2, ensure_ascii=False), language="json")
        api = get_client().api_url or "http://127.0.0.1:8000"
        st.caption("Para testar no terminal:")
        st.code(
            f"curl -X POST {api}/predict \\\n  -H 'Content-Type: application/json' \\\n"
            f"  -d '{json.dumps(payload, ensure_ascii=False)}'",
            language="bash",
        )


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def render() -> None:
    _init_state()
    components.page_header(
        "Análise individual",
        "Este cliente vai cancelar?",
        "Monte um perfil (ou carregue um exemplo) e veja, na hora, a probabilidade de "
        "cancelamento, se vale a pena agir, o que mais pesou na previsão e quais ofertas "
        "mudariam o risco.",
    )

    with st.container(key="analysis"):
        left, right = st.columns([5, 7], gap="large")
        with left:
            _form()
        with right:
            payload = _payload()
            errors = consistency_errors(payload)
            if errors:
                components.note(
                    "<b>Combinação impossível:</b> "
                    + esc("; ".join(errors))
                    + ". Ajuste o perfil ao lado.",
                    kind="warn",
                )
                return
            try:
                with st.spinner("Calculando…"):
                    result = _analyze(payload)
            except InvalidCustomerError:
                components.note(
                    "Os dados deste perfil foram recusados pela validação.", kind="warn"
                )
                return
            _verdict(payload, result)
            p = result["prediction"]["churn_probability"]
            test = test_predictions()
            if len(test):
                components.section(
                    "Contexto",
                    "Onde este cliente está na base",
                    f"Previsões do modelo para os {fmt_int(len(test))} clientes de teste, que ele "
                    "não viu no treino.",
                )
                components.card(
                    charts.risk_scale(
                        test,
                        evaluation().get("risk_level_cuts", {"baixo_max": 0.5, "medio_max": 0.5}),
                        marker=p,
                    )
                )
            _explanation(payload, result)
            _scenarios(payload, result)
            st.write("")
            _api_details(payload, result)