"""Treino do modelo de churn com rastreamento no MLflow.

Rode a partir da raiz do projeto: `python -m src.train`
"""

import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.utils import TARGET_COL, preprocess_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

N_ESTIMATORS = 100
MAX_DEPTH = 10
CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42


def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega e limpa o dataset bruto.

    `TotalCharges` vem do CSV como texto e usa string vazia (não NaN) para os
    valores faltantes. A coerção numérica precisa vir ANTES do dropna — na
    ordem inversa o dropna não encontra nulo nenhum e as 11 linhas
    problemáticas entram no treino silenciosamente.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    df = df.dropna(subset=["TotalCharges"])
    print(f"Linhas descartadas por TotalCharges ausente: {n_missing}")

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("churn-prediction")

    df_raw = load_raw_data(RAW_DATA_PATH)
    df_model = preprocess_features(df_raw)

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

    with mlflow.start_run(run_name="random_forest_v1"):
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print("\n=== Resultados ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("\n", classification_report(y_test, y_pred))

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "class_weight": CLASS_WEIGHT,
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, y_pred),
            input_example=X_train.head(1),
        )

        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
        joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))

        importances = sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "top_features": [
                        {"feature": name, "importance": round(float(value), 4)}
                        for name, value in importances[:8]
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print("\nModelo salvo em models/churn_model.pkl")
    print("Experimento registrado no MLflow")


if __name__ == "__main__":
    main()