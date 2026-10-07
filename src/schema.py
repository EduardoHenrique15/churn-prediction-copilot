"""Contrato de dados da API (Pydantic).

Fica fora de `src/api.py` para ser reaproveitado sem o FastAPI: o preditor
local (`src/predictor.py`), usado pela interface e pelo agente enquanto a API
hiberna, valida os clientes com EXATAMENTE as mesmas regras da API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from src.utils import (
    NUMERIC_LIMITS,
    ContractType,
    Gender,
    InternetServiceType,
    PaymentMethodType,
    YesNo,
    YesNoInternet,
    YesNoPhone,
    consistency_errors,
)

# Teto de clientes por chamada de /predict/batch: a previsão é vetorizada e
# barata; o limite existe para uma requisição não segurar o único worker do
# free tier nem aceitar um JSON arbitrariamente grande.
MAX_BATCH_SIZE = 1000

EXAMPLE_CUSTOMER = {
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


def _limit(field: str) -> dict:
    low, high = NUMERIC_LIMITS[field]
    return {"ge": low, "le": high}


class CustomerData(BaseModel):
    """Perfil do cliente. Campos categóricos só aceitam os valores do
    dataset de treino, e combinações impossíveis (sem internet, mas com
    suporte técnico; total gasto incompatível com tempo × mensalidade) são
    recusadas com 422 — o modelo nunca viu esses clientes."""

    gender: Gender
    SeniorCitizen: int = Field(**_limit("SeniorCitizen"))
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(**_limit("tenure"), description="Meses de contrato")
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
    MonthlyCharges: float = Field(**_limit("MonthlyCharges"))
    TotalCharges: float = Field(**_limit("TotalCharges"))

    model_config = {"json_schema_extra": {"example": EXAMPLE_CUSTOMER}}

    @model_validator(mode="after")
    def _check_consistency(self) -> CustomerData:
        errors = consistency_errors(self.model_dump())
        if errors:
            raise ValueError("; ".join(errors))
        return self


class PredictionResponse(BaseModel):
    # churn_prediction = a probabilidade passou do threshold de decisão, ou
    # seja, CONTATAR este cliente tem valor esperado positivo.
    churn_prediction: bool
    churn_probability: float
    risk_level: str


class BatchPredictRequest(BaseModel):
    customers: list[CustomerData] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    """Contribuição de cada variável para ESTE cliente.

    base_value + soma(shap_value) = saída do modelo no espaço `link`:
    "logit" (log-odds; aplique a sigmoide para obter a probabilidade) ou
    "identity" (já é probabilidade)."""

    base_value: float
    churn_probability: float
    link: str = "logit"
    contributions: list[FeatureContribution]
