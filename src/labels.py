"""Nomes em português para campos, categorias e features do modelo.

O modelo trabalha com nomes técnicos em inglês ("Contract_Two year",
"InternetService_Fiber optic") — os valores do dataset original, que NÃO
podem ser traduzidos no payload (o encoding quebraria). Este módulo traduz
só na hora de mostrar: interface, explicações e respostas do agente.
"""

from __future__ import annotations

from src.business import fmt_brl

_SIM_NAO = {"Yes": "Sim", "No": "Não"}
_SEM_INTERNET = {**_SIM_NAO, "No internet service": "Sem internet"}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "gender": {"Female": "Feminino", "Male": "Masculino"},
    "MultipleLines": {**_SIM_NAO, "No phone service": "Sem telefone"},
    "InternetService": {"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sem internet"},
    "Contract": {
        "Month-to-month": "Mensal",
        "One year": "Anual (1 ano)",
        "Two year": "Bienal (2 anos)",
    },
    "PaymentMethod": {
        "Electronic check": "Cheque eletrônico",
        "Mailed check": "Cheque por correio",
        "Bank transfer (automatic)": "Transferência bancária (automática)",
        "Credit card (automatic)": "Cartão de crédito (automático)",
    },
    **dict.fromkeys(("Partner", "Dependents", "PhoneService", "PaperlessBilling"), _SIM_NAO),
    **dict.fromkeys(
        (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ),
        _SEM_INTERNET,
    ),
}

FIELD_LABELS: dict[str, str] = {
    "gender": "Gênero",
    "SeniorCitizen": "Idoso",
    "Partner": "Cônjuge",
    "Dependents": "Dependentes",
    "tenure": "Tempo de casa",
    "PhoneService": "Telefone",
    "MultipleLines": "Múltiplas linhas",
    "InternetService": "Internet",
    "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online",
    "DeviceProtection": "Proteção de aparelho",
    "TechSupport": "Suporte técnico",
    "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes",
    "Contract": "Contrato",
    "PaperlessBilling": "Fatura digital",
    "PaymentMethod": "Pagamento",
    "MonthlyCharges": "Mensalidade",
    "TotalCharges": "Total gasto",
}


def value_label(field: str, value) -> str:
    """Rótulo em português de um valor cru ("Fiber optic" -> "Fibra óptica")."""
    if field == "SeniorCitizen":
        return "Sim" if int(value) == 1 else "Não"
    return VALUE_LABELS.get(field, {}).get(value, str(value))


def _split_feature(feature: str) -> tuple[str, str | None]:
    """'Contract_Two year' -> ('Contract', 'Two year'); numéricas -> (nome, None)."""
    if feature in FIELD_LABELS:
        return feature, None
    field, _, category = feature.partition("_")
    return field, category


def _category_label(field: str, category: str) -> str:
    return VALUE_LABELS.get(field, {}).get(category, category).lower()


def feature_name(feature: str) -> str:
    """Nome legível de uma feature do modelo, sem valor.

    'Contract_Two year' -> 'Contrato: bienal (2 anos)'
    """
    field, category = _split_feature(feature)
    label = FIELD_LABELS.get(field, field)
    if category is None or category == "Yes":
        return label
    if field == "InternetService" and category == "No":
        return "Sem internet"
    if field == "gender":
        return "Gênero masculino" if category == "Male" else label
    return f"{label}: {_category_label(field, category)}"


def describe_feature(feature: str, value: float) -> str:
    """Feature + valor deste cliente em português, para as explicações.

    Numéricas mostram o valor real; as demais são dummies do one-hot (0/1),
    então a frase diz se o cliente TEM ou NÃO TEM aquela categoria:
    'Contrato: não é bienal (2 anos)' em vez de 'Contract_Two year = 0'.
    """
    field, category = _split_feature(feature)
    label = FIELD_LABELS.get(field, field)
    if field == "tenure":
        return f"{label}: {value:.0f} {'mês' if round(value) == 1 else 'meses'}"
    if field in ("MonthlyCharges", "TotalCharges"):
        return f"{label}: {fmt_brl(value, cents=True)}"

    active = value >= 0.5
    if field == "SeniorCitizen" or category == "Yes":
        return f"{label}: {'sim' if active else 'não'}"
    if field == "gender":
        return f"{label}: {'masculino' if active else 'feminino'}"
    if field == "InternetService" and category == "No":
        return "Sem internet" if active else "Tem internet"

    cat = _category_label(field, category)
    if cat.startswith("sem "):
        # "No internet service" / "No phone service": a negação natural de
        # "sem internet" é "com internet", não "não é sem internet".
        return f"{label}: {cat}" if active else f"{label}: com {cat[4:]}"
    return f"{label}: {cat}" if active else f"{label}: não é {cat}"


def field_value_text(field: str, value) -> str:
    """Valor cru de um campo, formatado para leitura ('Fiber optic' -> 'Fibra óptica')."""
    if field == "tenure":
        months = int(round(float(value)))
        return f"{months} {'mês' if months == 1 else 'meses'}"
    if field in ("MonthlyCharges", "TotalCharges"):
        return fmt_brl(float(value), cents=True)
    return value_label(field, value)


def field_contributions(contributions: list[dict], record: dict) -> list[dict]:
    """Soma as contribuições das dummies de cada campo original.

    O modelo enxerga 'Contract_One year' e 'Contract_Two year' separados; para
    quem lê, o que importa é "Contrato: Mensal" com o efeito somado das duas
    colunas. Para a regressão logística a soma é exata (as contribuições são
    aditivas em log-odds). Ordenado da maior para a menor influência.
    """
    totals: dict[str, float] = {}
    for c in contributions:
        field, _ = _split_feature(c["feature"])
        totals[field] = totals.get(field, 0.0) + float(c["shap_value"])
    rows = [
        {
            "field": field,
            "label": FIELD_LABELS.get(field, field),
            "value": field_value_text(field, record[field]) if field in record else "",
            "contribution": total,
        }
        for field, total in totals.items()
    ]
    return sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)
