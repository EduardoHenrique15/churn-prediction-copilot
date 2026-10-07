"""Testes de src/utils.py — o schema e o encoding compartilhados.

O bug mais grave da versão anterior morava aqui: `pd.get_dummies` sobre UM
cliente descartava todas as dummies, e todo cliente virava a categoria de
referência (contrato mensal, DSL, transferência...). Estes testes travam o
encoding com categorias fixas para esse bug não voltar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    CATEGORICAL_COLS,
    CATEGORY_OPTIONS,
    NUMERIC_LIMITS,
    RAW_INPUT_COLUMNS,
    align_columns,
    as_number,
    consistency_errors,
    expected_total_charges,
    preprocess_features,
)
from tests.conftest import NO_INTERNET_SERVICES, loyal_customer, make_customer

EXAMPLE_BATCH = Path(__file__).resolve().parent.parent / "data" / "exemplo_lote.csv"


class TestPreprocessFeatures:
    def test_um_cliente_sozinho_gera_todas_as_dummies(self):
        """A regressão do bug: com categorias fixas, uma linha só gera as
        mesmas colunas que o dataset inteiro gera."""
        encoded = preprocess_features(pd.DataFrame([make_customer()]))
        for col, options in CATEGORY_OPTIONS.items():
            for category in sorted(options)[1:]:
                assert f"{col}_{category}" in encoded.columns

    def test_categoria_do_cliente_vira_1_na_dummy_certa(self):
        encoded = preprocess_features(
            pd.DataFrame([make_customer(Contract="Two year", InternetService="Fiber optic")])
        )
        assert encoded["Contract_Two year"].iloc[0] == 1
        assert encoded["Contract_One year"].iloc[0] == 0
        assert encoded["InternetService_Fiber optic"].iloc[0] == 1

    def test_categoria_de_referencia_zera_todas_as_dummies_do_campo(self):
        """drop_first=True: a primeira categoria em ordem alfabética
        ('Month-to-month') é a referência e não tem coluna própria."""
        encoded = preprocess_features(pd.DataFrame([make_customer(Contract="Month-to-month")]))
        assert "Contract_Month-to-month" not in encoded.columns
        assert encoded[["Contract_One year", "Contract_Two year"]].sum(axis=1).iloc[0] == 0

    def test_encoding_de_um_cliente_nao_depende_dos_vizinhos(self):
        a = make_customer(Contract="Two year")
        b = make_customer(Contract="One year", PaymentMethod="Mailed check")
        alone = preprocess_features(pd.DataFrame([a]))
        together = preprocess_features(pd.DataFrame([a, b]))
        pd.testing.assert_series_equal(
            alone.iloc[0], together.iloc[0], check_names=False, check_dtype=False
        )

    def test_categoria_fora_do_dominio_levanta_erro_em_vez_de_virar_referencia(self):
        with pytest.raises(ValueError, match="Contract"):
            preprocess_features(pd.DataFrame([make_customer(Contract="Lifetime")]))

    def test_numericas_viram_numero_mesmo_vindo_como_texto(self):
        encoded = preprocess_features(pd.DataFrame([make_customer(TotalCharges="1234.5")]))
        assert pd.api.types.is_numeric_dtype(encoded["TotalCharges"])
        assert encoded["TotalCharges"].iloc[0] == 1234.5

    def test_valor_numerico_invalido_vira_nan_nao_erro(self):
        """O treino descarta essas linhas; a API barra antes, no Pydantic."""
        encoded = preprocess_features(pd.DataFrame([make_customer(TotalCharges=" ")]))
        assert encoded["TotalCharges"].isna().iloc[0]


class TestAlignColumns:
    def test_preserva_a_ordem_exata_pedida(self):
        """O scikit-learn casa features por POSIÇÃO, não por nome."""
        encoded = preprocess_features(pd.DataFrame([make_customer()]))
        order = ["TotalCharges", "SeniorCitizen", "tenure", "Contract_Two year"]
        assert list(align_columns(encoded, order).columns) == order

    def test_preenche_colunas_ausentes_com_zero_e_descarta_extras(self):
        df = pd.DataFrame([{"tenure": 3, "customerID": "abc"}])
        aligned = align_columns(df, ["tenure", "Contract_Two year"])
        assert list(aligned.columns) == ["tenure", "Contract_Two year"]
        assert aligned["Contract_Two year"].iloc[0] == 0


class TestSchema:
    def test_category_options_cobre_as_colunas_categoricas(self):
        assert set(CATEGORY_OPTIONS) == set(CATEGORICAL_COLS)

    @pytest.mark.parametrize("field", list(CATEGORY_OPTIONS))
    def test_cada_campo_tem_pelo_menos_duas_opcoes(self, field):
        assert len(CATEGORY_OPTIONS[field]) >= 2

    def test_raw_input_columns_bate_com_customerdata(self):
        """Alimenta a validação do CSV em lote: se a API ganhar um campo e a
        lista não acompanhar, o upload aceitaria arquivos incompletos."""
        from src.schema import CustomerData

        assert list(CustomerData.model_fields) == RAW_INPUT_COLUMNS

    @pytest.mark.parametrize("field", list(NUMERIC_LIMITS))
    def test_limites_numericos_batem_com_customerdata(self, field):
        from src.schema import CustomerData

        metadata = CustomerData.model_fields[field].metadata
        ge = next(m.ge for m in metadata if hasattr(m, "ge"))
        le = next(m.le for m in metadata if hasattr(m, "le"))
        assert (ge, le) == NUMERIC_LIMITS[field]


class TestConsistencyErrors:
    def test_cliente_de_exemplo_e_consistente(self):
        assert consistency_errors(make_customer()) == []

    def test_cliente_sem_internet_consistente(self):
        assert consistency_errors(loyal_customer()) == []

    def test_sem_internet_mas_com_suporte_tecnico(self):
        record = make_customer(**{**NO_INTERNET_SERVICES, "TechSupport": "Yes"})
        errors = consistency_errors(record)
        assert len(errors) == 1 and "TechSupport" in errors[0]

    def test_com_internet_mas_no_internet_service(self):
        errors = consistency_errors(make_customer(OnlineBackup="No internet service"))
        assert errors and "OnlineBackup" in errors[0]

    def test_sem_telefone_mas_com_multiplas_linhas(self):
        errors = consistency_errors(make_customer(PhoneService="No", MultipleLines="Yes"))
        assert errors and "MultipleLines" in errors[0]

    def test_com_telefone_mas_no_phone_service(self):
        assert consistency_errors(make_customer(MultipleLines="No phone service"))

    def test_sem_telefone_e_sem_internet_nao_existe(self):
        record = {
            **loyal_customer(),
            "PhoneService": "No",
            "MultipleLines": "No phone service",
        }
        assert any("telefone, internet" in e for e in consistency_errors(record))

    @pytest.mark.parametrize("total", [100.0, 5000.0])
    def test_total_gasto_incompativel_com_tempo_e_mensalidade(self, total):
        # 5 meses × R$ 85,50 ≈ R$ 427,50: R$ 100 e R$ 5.000 estão fora da faixa.
        errors = consistency_errors(make_customer(TotalCharges=total))
        assert errors and "TotalCharges" in errors[0]

    def test_tolera_reajuste_e_desconto_realistas(self):
        assert consistency_errors(make_customer(TotalCharges=427.5 * 1.3)) == []
        assert consistency_errors(make_customer(TotalCharges=427.5 * 0.7)) == []

    def test_cliente_novo_com_total_zero_e_valido(self):
        assert consistency_errors(make_customer(tenure=0, TotalCharges=0.0)) == []

    def test_aceita_tipos_do_numpy(self):
        """Linhas de um DataFrame chegam como np.int64/np.float64."""
        record = make_customer(tenure=np.int64(5), MonthlyCharges=np.float64(85.5))
        record["TotalCharges"] = np.float64(100.0)
        assert consistency_errors(record)

    def test_clientes_reais_do_exemplo_passam_em_todas_as_regras(self):
        df = pd.read_csv(EXAMPLE_BATCH)
        violations = [i for i, row in df.iterrows() if consistency_errors(row.to_dict())]
        assert violations == []


class TestHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (3, 3.0),
            ("4.5", 4.5),
            (np.int64(7), 7.0),
            (None, None),
            ("abc", None),
            (float("nan"), None),
        ],
    )
    def test_as_number(self, value, expected):
        assert as_number(value) == expected

    def test_as_number_nao_trata_booleano_como_numero(self):
        assert as_number(True) is None

    def test_expected_total_charges(self):
        assert expected_total_charges(12, 70.25) == 843.0
