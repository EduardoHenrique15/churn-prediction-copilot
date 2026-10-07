"""Explicação por previsão: quanto cada variável empurrou o risco deste cliente.

Dois caminhos, conforme o modelo escolhido pelo treino:

- Regressão logística (pipeline scaler + LogisticRegression): a explicação é
  EXATA e não precisa de biblioteca nenhuma. O log-odds é
  intercepto + Σ coef_i × z_i, com z = (x − média) / desvio do treino; a
  contribuição de cada variável é coef_i × z_i e o ponto de partida é o
  intercepto (o log-odds de um cliente com todas as variáveis na média).
  Para modelos lineares isso coincide com os valores SHAP.
- Modelos de árvore (Random Forest, Gradient Boosting): SHAP TreeExplainer,
  importado só na primeira chamada — o import sozinho soma ~100 MB de RAM.

Nos dois casos vale: ponto de partida + soma das contribuições = saída do
modelo, no espaço indicado por `link` ("logit" = log-odds, "identity" =
probabilidade).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Explanation:
    base_value: float
    contributions: np.ndarray
    link: str  # "logit" ou "identity"


def is_linear(model) -> bool:
    return isinstance(model, Pipeline) and hasattr(model.named_steps.get("clf"), "coef_")


def _explain_linear(model: Pipeline, X: pd.DataFrame) -> Explanation:
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]
    z = (X.to_numpy(dtype=float) - scaler.mean_) / scaler.scale_
    contributions = clf.coef_[0] * z[0]
    return Explanation(float(clf.intercept_[0]), contributions, "logit")


_tree_explainers: dict[int, object] = {}


def _explain_tree(model, X: pd.DataFrame) -> Explanation:
    import shap  # sob demanda: só modelos de árvore pagam o custo de memória

    key = id(model)
    if key not in _tree_explainers:
        _tree_explainers[key] = shap.TreeExplainer(model)
    explainer = _tree_explainers[key]

    values = explainer.shap_values(X)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    # Classificadores binários podem devolver uma lista por classe ou um
    # array (amostras, features, classes), dependendo do modelo e da versão
    # do shap — normalizamos para a classe positiva.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    base = expected[-1] if len(expected) > 1 else expected[0]

    # Gradient Boosting explica em log-odds; Random Forest, em probabilidade.
    link = "identity" if hasattr(model, "estimators_") else "logit"
    return Explanation(float(base), values[0], link)


def explain(model, X: pd.DataFrame) -> Explanation:
    """Explica UMA linha já codificada (mesmas colunas do treino)."""
    if len(X) != 1:
        raise ValueError("explain() recebe exatamente um cliente")
    if is_linear(model):
        return _explain_linear(model, X)
    return _explain_tree(model, X)