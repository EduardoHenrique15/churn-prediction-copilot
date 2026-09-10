import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
from utils import preprocess_features

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlflow.db"))
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

df_raw = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw = df_raw.drop(columns=["customerID"])
df_raw = df_raw.dropna(subset=["TotalCharges"])  
df_raw["Churn"] = df_raw["Churn"].map({"Yes": 1, "No": 0})

df_model = preprocess_features(df_raw)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="random_forest_v1"):

    n_estimators = 100
    max_depth = 10
    class_weight = "balanced"  

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Resultados ===")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\n", classification_report(y_test, y_pred))

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("class_weight", class_weight)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/churn_model.pkl")

    joblib.dump(list(X.columns), "../models/feature_columns.pkl")

    print("\n✅ Modelo salvo em models/churn_model.pkl")
    print("✅ Experimento registrado no MLflow")