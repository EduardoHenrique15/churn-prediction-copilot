"""API REST de previsão de churn (FastAPI).

Sobe com: `uvicorn src.api:app --reload` a partir da raiz do projeto.
"""

import logging
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import (
    ContractType,
    Gender,
    InternetServiceType,
    PaymentMethodType,
    YesNo,
    YesNoInternet,
    YesNoPhone,
    align_columns,
    preprocess_features,
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

RISK_THRESHOLD_HIGH = 0.65
RISK_THRESHOLD_MEDIUM = 0.35

try:
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Modelo não encontrado em {MODEL_PATH}. "
        "Rode `python -m src.train` antes de subir a API."
    ) from exc

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Prevê a probabilidade de cancelamento de um cliente de telecom "
        "usando um RandomForestClassifier treinado no dataset Telco Customer Churn."
    ),
    version="1.1.0",
)


class CustomerData(BaseModel):
    """Perfil do cliente. Os campos categóricos aceitam apenas os valores
    presentes no dataset de treino — qualquer outro valor é rejeitado com 422
    em vez de gerar uma previsão sobre dados que o modelo nunca viu."""

    gender: Gender
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Meses de contrato")
    PhoneService: YesNo
    MultipleLines: YesNoPhone
    InternetService: InternetServiceType
    OnlineSecurity: YesNoInternet
    OnlineBackup: YesNoInternet
    DeviceProtection: YesNoInternet
    TechSupport: YesNoInternet
    StreamingTV: YesNoInternet
    StreamingMovies: YesNoInternet
    Contract: ContractType
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodType
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100_000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 450.75,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    risk_level: str


def _risk_level(probability: float) -> str:
    if probability >= RISK_THRESHOLD_HIGH:
        return "Alto"
    if probability >= RISK_THRESHOLD_MEDIUM:
        return "Médio"
    return "Baixo"


@app.get("/")
def root() -> dict:
    return {"status": "online", "message": "Churn Prediction API está no ar"}


@app.get("/health")
def health() -> dict:
    """Usado pelo healthcheck do Render e pela interface para detectar
    se o serviço já saiu da hibernação do free tier."""
    return {"status": "healthy", "model_features": len(expected_columns)}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData) -> PredictionResponse:
    df_input = pd.DataFrame([data.model_dump()])

    try:
        df_final = align_columns(preprocess_features(df_input), expected_columns)
        probability = float(model.predict_proba(df_final)[0][1])
    except Exception:
        logger.exception("Falha ao gerar previsão")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar a previsão.")

    return PredictionResponse(
        churn_prediction=probability >= 0.5,
        churn_probability=round(probability, 4),
        risk_level=_risk_level(probability),
    )