from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from utils import preprocess_features, align_columns    

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
expected_columns = joblib.load(COLUMNS_PATH)

app = FastAPI(
    title="Churn Prediction API",
    description="API para prever cancelamento de clientes (churn) usando RandomForest",
    version="1.0.0"
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
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
                "TotalCharges": 450.75
            }
        }

@app.get("/")
def root():
    return {"status": "online", "message": "Churn Prediction API está no ar"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])

        df_encoded = preprocess_features(df_input)
        df_final = align_columns(df_encoded, expected_columns)

        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0][1]

        return {
            "churn_prediction": bool(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": "Alto" if probability >= 0.5 else "Baixo"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar previsão: {str(e)}")