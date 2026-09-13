"""Pré-processamento e schema de features compartilhados.

Este módulo é a fonte única de verdade sobre o formato dos dados de entrada:
- `train.py` usa `preprocess_features` para montar a matriz de treino;
- `api.py` usa os mesmos tipos para validar o payload e o mesmo
  `preprocess_features` para transformar a requisição;
- `app.py` usa `CATEGORY_OPTIONS` para montar os selects da interface.

Manter isso centralizado é o que impede o clássico train/serve skew: se a UI
oferecesse uma opção que o modelo nunca viu, a previsão sairia silenciosamente
errada em vez de falhar.
"""

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


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o mesmo pré-processamento no treino e na inferência.

    Converte as colunas numéricas para tipo numérico (valores inválidos viram
    NaN) e aplica one-hot encoding com `drop_first=True` nas categóricas.
    """
    df = df.copy()

    for col in ("TotalCharges", "MonthlyCharges"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def align_columns(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Reindexa o DataFrame para exatamente as colunas que o modelo espera.

    Uma requisição individual gera apenas as dummies das categorias presentes
    nela; as demais precisam existir preenchidas com 0, na ordem do treino.
    """
    return df.reindex(columns=expected_columns, fill_value=0)