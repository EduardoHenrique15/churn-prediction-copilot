import pandas as pd

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o mesmo pré-processamento usado no treino:
    - Garante que colunas numéricas estejam no tipo correto
    - Aplica one-hot encoding nas colunas categóricas

    Usado tanto no treino (train.py) quanto na API (api.py),
    garantindo que os dados sejam sempre transformados da mesma forma.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    return df


def align_columns(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Garante que o DataFrame final tenha exatamente as colunas que o modelo espera,
    na mesma ordem, preenchendo com 0 qualquer coluna que não tenha sido gerada
    (ex: categoria que não apareceu num input específico da API).
    """
    return df.reindex(columns=expected_columns, fill_value=0)