"""Schema e pré-processamento compartilhados entre treino, API e interface.

Este módulo é a fonte única de verdade sobre o formato dos dados de entrada:
- `train.py` usa `preprocess_features` para montar a matriz de treino;
- `api.py` usa os mesmos tipos para validar o payload e o mesmo
  `preprocess_features` para transformar a requisição;
- a interface usa `CATEGORY_OPTIONS` e as regras de consistência para montar
  o formulário e validar o CSV em lote.

Centralizar isso é o que impede o train/serve skew: a mesma função codifica
um cliente no treino e na previsão, com as MESMAS colunas e os MESMOS valores.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, get_args

import pandas as pd

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
YesNoPhone = Literal["Yes", "No", "No phone service"]
YesNoInternet = Literal["Yes", "No", "No internet service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

CATEGORY_OPTIONS: dict[str, tuple[str, ...]] = {
    "gender": get_args(Gender),
    "Partner": get_args(YesNo),
    "Dependents": get_args(YesNo),
    "PhoneService": get_args(YesNo),
    "MultipleLines": get_args(YesNoPhone),
    "InternetService": get_args(InternetServiceType),
    "OnlineSecurity": get_args(YesNoInternet),
    "OnlineBackup": get_args(YesNoInternet),
    "DeviceProtection": get_args(YesNoInternet),
    "TechSupport": get_args(YesNoInternet),
    "StreamingTV": get_args(YesNoInternet),
    "StreamingMovies": get_args(YesNoInternet),
    "Contract": get_args(ContractType),
    "PaperlessBilling": get_args(YesNo),
    "PaymentMethod": get_args(PaymentMethodType),
}

CATEGORICAL_COLS: list[str] = list(CATEGORY_OPTIONS)

NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"

# Serviços que só existem para quem tem internet. No dataset, quem não tem
# internet aparece SEMPRE com "No internet service" nesses 6 campos (0
# exceções em 7.043 clientes) — ver consistency_errors().
INTERNET_SERVICES: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

# As 19 colunas cruas que a API espera por cliente (iguais aos campos de
# CustomerData em api.py, nesta ordem). Um teste trava esta lista contra
# CustomerData.model_fields para as duas nunca divergirem.
RAW_INPUT_COLUMNS: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Tolerância de TotalCharges em relação a tenure × MonthlyCharges. No dataset
# a razão entre os dois fica entre 0,92 e 1,08 para 90% dos clientes (p5–p95);
# a tolerância é bem mais larga de propósito — o objetivo é barrar erro de
# digitação grosseiro (tempo de casa de 70 meses com total gasto de R$ 100),
# não recusar clientes reais que tiveram reajuste ou desconto.
TOTAL_CHARGES_RATIO_RANGE: tuple[float, float] = (0.5, 1.6)

# Limites numéricos aceitos pela API (os mesmos de CustomerData em
# src/schema.py — um teste trava os dois juntos). A interface usa a faixa
# observada no dataset, mais estreita, nos controles do formulário.
NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 100_000),
}


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica clientes para o modelo — igual no treino e na previsão.

    As categóricas viram `pd.Categorical` com as categorias FIXAS de
    `CATEGORY_OPTIONS` (em ordem alfabética, a mesma que o `get_dummies`
    usava sobre o dataset inteiro) antes do one-hot com `drop_first=True`.

    Por que isso importa: sem categorias fixas, `get_dummies` só enxerga os
    valores presentes NO DataFrame recebido. Com um único cliente, cada
    coluna tem uma categoria só — ela é "a primeira" e é descartada, e o
    cliente vira silenciosamente a categoria de referência em todas as 15
    variáveis (contrato mensal, DSL, transferência bancária...). Foi
    exatamente esse bug que existiu na versão anterior da API.

    Levanta ValueError se aparecer uma categoria fora do domínio, em vez de
    deixá-la virar a categoria de referência sem aviso.
    """
    df = df.copy()

    for col in ("TotalCharges", "MonthlyCharges"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in present:
        categories = sorted(CATEGORY_OPTIONS[col])
        unknown = df[col].notna() & ~df[col].isin(categories)
        if unknown.any():
            bad = sorted(df.loc[unknown, col].astype(str).unique())
            raise ValueError(f"Valor fora do domínio em '{col}': {bad}")
        df[col] = pd.Categorical(df[col], categories=categories)

    return pd.get_dummies(df, columns=present, drop_first=True, dtype=float)


def align_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Reindexa para exatamente as colunas (e a ordem) que o modelo espera.

    Com categorias fixas, `preprocess_features` já gera todas as dummies; o
    reindex garante a ordem do treino e descarta colunas extras que um CSV
    em lote possa trazer (customerID, Churn...).
    """
    return df.reindex(columns=expected_columns, fill_value=0)


def expected_total_charges(tenure: float, monthly_charges: float) -> float:
    """Estimativa de TotalCharges usada pela interface para preencher o campo
    automaticamente (no dataset, TotalCharges ≈ tenure × MonthlyCharges)."""
    return round(float(tenure) * float(monthly_charges), 2)


def as_number(value) -> float | None:
    """Número finito ou None — aceita int, float, tipos do numpy e texto."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def consistency_errors(record: Mapping) -> list[str]:
    """Combinações que não existem no mundo real (nem no dataset).

    Os valores de cada campo podem ser válidos isoladamente e o cliente
    ainda assim ser impossível — sem internet, mas com suporte técnico. O
    modelo nunca viu esses casos e devolveria uma previsão sem sentido; a
    API recusa (422) e a interface impede que eles sejam montados.

    Devolve a lista de problemas em português (vazia = cliente consistente).
    """
    errors: list[str] = []

    internet = record.get("InternetService")
    if internet == "No":
        wrong = [f for f in INTERNET_SERVICES if record.get(f) != "No internet service"]
        if wrong:
            errors.append(
                "Cliente sem internet precisa ter 'No internet service' em: " + ", ".join(wrong)
            )
    elif internet is not None:
        wrong = [f for f in INTERNET_SERVICES if record.get(f) == "No internet service"]
        if wrong:
            errors.append(
                "Cliente com internet não pode ter 'No internet service' em: " + ", ".join(wrong)
            )

    phone = record.get("PhoneService")
    multiple = record.get("MultipleLines")
    if phone == "No" and multiple not in (None, "No phone service"):
        errors.append("Cliente sem telefone precisa ter MultipleLines = 'No phone service'")
    if phone == "Yes" and multiple == "No phone service":
        errors.append("Cliente com telefone não pode ter MultipleLines = 'No phone service'")
    # Nenhum dos 7.043 clientes do dataset fica sem os dois serviços.
    if phone == "No" and internet == "No":
        errors.append("Cliente precisa ter telefone, internet ou os dois")

    tenure = as_number(record.get("tenure"))
    monthly = as_number(record.get("MonthlyCharges"))
    total = as_number(record.get("TotalCharges"))
    if None not in (tenure, monthly, total) and tenure > 0:
        expected = tenure * monthly
        low, high = TOTAL_CHARGES_RATIO_RANGE
        if expected > 0 and not (low * expected <= total <= high * expected):
            errors.append(
                f"TotalCharges ({total:.2f}) incompatível com tenure × MonthlyCharges "
                f"(≈ {expected:.2f})"
            )

    return errors