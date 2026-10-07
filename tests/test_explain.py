"""Testes de src/explain.py — a explicação soma exatamente a previsão."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explain import explain, is_linear
from src.predictor import ChurnPredictor
from tests.conftest import loyal_customer, make_customer


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(400, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + 0.5 * X["b"] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    return X, y


def test_linear_reproduz_o_log_odds_do_modelo(synthetic):
    X, y = synthetic
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y)
    assert is_linear(model)
    for i in range(5):
        row = X.iloc[[i]]
        result = explain(model, row)
        p = model.predict_proba(row)[0, 1]
        assert result.link == "logit"
        assert result.base_value + result.contributions.sum() == pytest.approx(_logit(p), abs=1e-9)


def test_modelo_publicado_explica_clientes_reais_de_forma_exata():
    predictor = ChurnPredictor()
    for record in (make_customer(), loyal_customer()):
        body = predictor.explain(record)
        total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
        assert 1 / (1 + math.exp(-total)) == pytest.approx(body["churn_probability"], abs=1e-4)


@pytest.mark.parametrize(
    "model,link",
    [
        (HistGradientBoostingClassifier(max_iter=30, random_state=0), "logit"),
        (RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0), "identity"),
    ],
)
def test_arvores_usam_shap_e_somam_a_saida_no_espaco_certo(synthetic, model, link):
    pytest.importorskip("shap")
    X, y = synthetic
    model.fit(X, y)
    assert not is_linear(model)
    row = X.iloc[[3]]
    result = explain(model, row)
    p = model.predict_proba(row)[0, 1]
    expected = _logit(p) if link == "logit" else p
    assert result.link == link
    assert result.base_value + result.contributions.sum() == pytest.approx(expected, abs=1e-4)


def test_explica_exatamente_um_cliente(synthetic):
    X, y = synthetic
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y)
    with pytest.raises(ValueError):
        explain(model, X.iloc[:2])