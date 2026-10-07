"""Testes de src/labels.py — a tradução do modelo para o português."""

import pytest

from src.labels import (
    FIELD_LABELS,
    VALUE_LABELS,
    describe_feature,
    feature_name,
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


@pytest.mark.parametrize(
    "feature,expected",
    [
        ("Contract_Two year", "Contrato: bienal (2 anos)"),
        ("InternetService_No", "Sem internet"),
        ("gender_Male", "Gênero masculino"),
        ("TechSupport_Yes", "Suporte técnico"),
        ("tenure", "Tempo de casa"),
    ],
)
def test_feature_name(feature, expected):
    assert feature_name(feature) == expected


@pytest.mark.parametrize(
    "feature,value,expected",
    [
        ("tenure", 1, "Tempo de casa: 1 mês"),
        ("tenure", 5, "Tempo de casa: 5 meses"),
        ("MonthlyCharges", 85.5, "Mensalidade: R$ 85,50"),
        ("Contract_Two year", 0, "Contrato: não é bienal (2 anos)"),
        ("OnlineSecurity_No internet service", 0, "Segurança online: com internet"),
        ("SeniorCitizen", 1, "Idoso: sim"),
    ],
)
def test_describe_feature(feature, value, expected):
    assert describe_feature(feature, value) == expected


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
