"""Núcleo de previsão: carrega os artefatos do treino e aplica a decisão.

É o mesmo código nos dois caminhos do projeto:
- a API (`src/api.py`) é uma camada HTTP fina sobre este módulo;
- a interface e o agente usam este módulo como plano B enquanto a API do
  Render hiberna (ver `src/client.py`).

Encoding, threshold e faixas de risco moram aqui — a previsão de um cliente
é idêntica venha da API ou do cálculo local.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.business import risk_cuts, risk_level
from src.explain import explain as explain_model
from src.explain import is_linear
from src.utils import align_columns, preprocess_features

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class ChurnPredictor:
    """Modelo + regras de decisão derivadas do treino."""

    def __init__(self, models_dir: str | Path = MODELS_DIR):
        models_dir = Path(models_dir)
        model_path = models_dir / "churn_model.pkl"
        try:
            self.model = joblib.load(model_path)
            self.columns: list[str] = list(joblib.load(models_dir / "feature_columns.pkl"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Modelo não encontrado em {model_path}. Rode `python -m src.train` antes."
            ) from exc

        evaluation = _read_json(models_dir / "evaluation.json")
        if "optimal_threshold" in evaluation and "risk_level_cuts" in evaluation:
            # Threshold e faixas vêm do treino (ver src/train.py): o threshold
            # maximiza o valor líquido de uma política de retenção nas
            # previsões fora-da-amostra do treino.
            self.threshold = float(evaluation["optimal_threshold"])
            self.cuts = {k: float(v) for k, v in evaluation["risk_level_cuts"].items()}
        else:
            logger.warning(
                "models/evaluation.json ausente ou incompleto — usando threshold 0,5. "
                "Rode `python -m src.train` para gerar os valores derivados dos dados."
            )
            self.threshold = 0.5
            self.cuts = risk_cuts(0.5)
        self.model_selected: str = evaluation.get("model_selected", "desconhecido")
        self.trained_at: str | None = evaluation.get("trained_at")

    @property
    def is_linear(self) -> bool:
        return is_linear(self.model)

    def encode(self, records: Sequence[Mapping]) -> pd.DataFrame:
        """Clientes crus -> matriz com as colunas (e a ordem) do treino."""
        return align_columns(preprocess_features(pd.DataFrame(list(records))), self.columns)

    def predict_proba(self, records: Sequence[Mapping]) -> np.ndarray:
        return self.model.predict_proba(self.encode(records))[:, 1]

    def decide(self, probability: float) -> dict:
        return {
            "churn_prediction": bool(probability >= self.threshold),
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level(float(probability), self.cuts),
        }

    def predict(self, records: Sequence[Mapping]) -> list[dict]:
        """Um único predict_proba para o lote inteiro. Com categorias fixas no
        encoding, cada cliente recebe o mesmo resultado que teria sozinho."""
        return [self.decide(p) for p in self.predict_proba(records)]

    def explain(self, record: Mapping) -> dict:
        """Contribuição de cada variável, ordenada por magnitude."""
        X = self.encode([record])
        result = explain_model(self.model, X)
        probability = float(self.model.predict_proba(X)[0][1])
        contributions = sorted(
            (
                {
                    "feature": col,
                    "feature_value": float(X[col].iloc[0]),
                    "shap_value": float(result.contributions[i]),
                }
                for i, col in enumerate(self.columns)
            ),
            key=lambda c: abs(c["shap_value"]),
            reverse=True,
        )
        return {
            "base_value": round(result.base_value, 6),
            "churn_probability": round(probability, 4),
            "link": result.link,
            "contributions": contributions,
        }