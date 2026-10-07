"""Testes de src/validation.py — validação linha a linha do CSV em lote."""

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.schema import CustomerData
from src.validation import check_row, prepare_batch
from tests.conftest import NO_INTERNET_SERVICES, make_customer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("name", ["exemplo_lote.csv", "exemplo_lote_novos.csv"])
def test_arquivos_de_exemplo_sao_100_por_cento_validos(name):
    df = pd.read_csv(DATA_DIR / name)
    clean, records, problems, notes = prepare_batch(df)
    assert len(records) == len(df)
    assert problems.empty
    assert notes == []


def test_linhas_com_problema_sao_listadas_com_o_numero_da_planilha():
    rows = [make_customer(), make_customer(Contract="Vitalício"), make_customer(tenure="abc")]
    clean, records, problems, notes = prepare_batch(pd.DataFrame(rows))
    assert len(records) == 1
    assert problems["linha"].tolist() == [3, 4]  # cabeçalho é a linha 1
    assert "Contrato" in problems["problemas"].iloc[0]
    assert "Tempo de casa" in problems["problemas"].iloc[1]


def test_mensagem_de_categoria_invalida_lista_os_valores_aceitos():
    check = check_row(make_customer(Contract="Lifetime"))
    assert check.record is None
    assert "Month-to-month, One year ou Two year" in check.errors[0]


def test_cliente_novo_com_total_vazio_vira_zero_com_aviso():
    check = check_row(make_customer(tenure=0, TotalCharges=" "))
    assert check.record["TotalCharges"] == 0.0
    assert check.fixes


def test_total_vazio_com_tempo_de_casa_e_erro():
    check = check_row(make_customer(TotalCharges=None))
    assert check.record is None
    assert any("Total gasto" in e for e in check.errors)


def test_colunas_de_resultado_anterior_sao_descartadas():
    df = pd.DataFrame([make_customer()])
    df["probabilidade_churn"] = 0.5
    df["risco"] = "Alto"
    clean, records, problems, notes = prepare_batch(df)
    assert "probabilidade_churn" not in clean.columns
    assert notes and "resultado anterior" in notes[0]


def test_falta_de_coluna_obrigatoria_e_erro_claro():
    df = pd.DataFrame([make_customer()]).drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Contract"):
        prepare_batch(df)


def test_espacos_em_nomes_e_valores_sao_tolerados():
    row = {f" {k} ": (f" {v} " if isinstance(v, str) else v) for k, v in make_customer().items()}
    clean, records, problems, notes = prepare_batch(pd.DataFrame([row]))
    assert len(records) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"gender": "Banana"},
        {"SeniorCitizen": 3},
        {"tenure": 130},
        {"tenure": 2.5},
        {"MonthlyCharges": -1},
        {"TotalCharges": 5000.0},
        {**NO_INTERNET_SERVICES, "StreamingTV": "Yes"},
        {"PhoneService": "No", "MultipleLines": "No"},
        {"PhoneService": "No", "MultipleLines": "No phone service"},
    ],
)
def test_concorda_com_a_validacao_da_api(overrides):
    """A interface não pode aceitar uma linha que a API recusaria (nem o
    contrário)."""
    record = make_customer(**overrides)
    try:
        CustomerData(**record)
        api_accepts = True
    except ValidationError:
        api_accepts = False
    assert (check_row(record).record is not None) == api_accepts