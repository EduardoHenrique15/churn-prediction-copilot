"""Testes de src/labels.py — a tradução do modelo para o português."""

import pytest

from src.labels import (
    FIELD_LABELS,
    VALUE_LABELS,
    field_contributions,
    field_value_text,
    value_label,
)
from src.utils import CATEGORY_OPTIONS, RAW_INPUT_COLUMNS
from tests.conftest import make_customer


def test_todo_campo_tem_rotulo_em_portugues():
    assert set(FIELD_LABELS) == set(RAW_INPUT_COLUMNS)


@pytest.mark.parametrize("field", list(CATEGORY_OPTIONS))
def test_todo_valor_categorico_tem_traducao(field):
    assert set(VALUE_LABELS[field]) == set(CATEGORY_OPTIONS[field])


def test_value_label():
    assert value_label("InternetService", "Fiber optic") == "Fibra óptica"
    assert value_label("SeniorCitizen", 0) == "Não"


def test_field_value_text():
    assert field_value_text("Contract", "Month-to-month") == "Mensal"
    assert field_value_text("tenure", 12.0) == "12 meses"
    assert field_value_text("TotalCharges", 1234.5) == "R$ 1.234,50"


def test_field_contributions_soma_as_dummies_de_cada_campo():
    contributions = [
        {"feature": "Contract_One year", "feature_value": 0.0, "shap_value": 0.2},
        {"feature": "Contract_Two year", "feature_value": 0.0, "shap_value": 0.3},
        {"feature": "tenure", "feature_value": 5.0, "shap_value": 0.9},
        {"feature": "gender_Male", "feature_value": 0.0, "shap_value": -0.01},
    ]
    rows = field_contributions(contributions, make_customer())
    assert [r["field"] for r in rows] == ["tenure", "Contract", "gender"]
    assert rows[1]["contribution"] == pytest.approx(0.5)
    assert rows[1]["value"] == "Mensal"
    assert rows[0]["value"] == "5 meses"


class TestModelNames:
    def test_nome_no_meio_da_frase_preserva_nomes_proprios(self):
        from src.labels import model_name

        assert model_name("logistic_regression") == "Regressão logística"
        assert model_name("logistic_regression", in_sentence=True) == "regressão logística"
        assert model_name("random_forest", in_sentence=True) == "Random Forest"
        assert model_name("desconhecido") == "desconhecido"

    def test_artigo_concorda_com_o_modelo(self):
        from src.labels import model_with_article

        assert model_with_article("logistic_regression") == "a regressão logística"
        assert (
            model_with_article("hist_gradient_boosting", capitalize=True) == "O Gradient Boosting"
        )
        assert (
            model_with_article("logistic_regression", indefinite=True) == "uma regressão logística"
        )

    def test_todo_candidato_do_treino_tem_nome(self):
        from src.labels import MODEL_NAMES
        from src.train import build_candidate_models

        assert set(build_candidate_models()) == set(MODEL_NAMES)
