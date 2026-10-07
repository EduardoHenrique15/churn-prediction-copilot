"""Testes de src/business.py — valor, threshold, faixas, cenários e drift."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.business import (
    PSI_SHIFT,
    PSI_STABLE,
    build_reference_profile,
    drift_report,
    expected_contact_value,
    fmt_brl,
    fmt_int,
    fmt_num,
    fmt_pct,
    fmt_pp,
    net_value,
    optimal_threshold,
    psi_status,
    risk_cuts,
    risk_level,
    theoretical_threshold,
    value_curve,
    whatif_scenarios,
)
from src.utils import CATEGORICAL_COLS, NUMERIC_COLS, consistency_errors
from tests.conftest import loyal_customer, make_customer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TestValor:
    def test_net_value_segue_a_formula(self):
        # 10 que iam cancelar × 30% × R$ 1.000 − 25 contatos × R$ 100
        assert net_value({"tp": 10, "fp": 15}, 1000, 100, 0.3) == pytest.approx(500.0)

    def test_value_curve_calcula_taxas(self):
        point = {"threshold": 0.3, "tp": 30, "fp": 20, "fn": 10, "tn": 40}
        (row,) = value_curve([point], 1000, 100, 0.3)
        assert row["contatados"] == 50
        assert row["taxa_contato"] == pytest.approx(0.5)
        assert row["recall"] == pytest.approx(0.75)
        assert row["precision"] == pytest.approx(0.6)
        assert row["valor"] == pytest.approx(30 * 300 - 50 * 100)

    def test_optimal_threshold_escolhe_o_maior_valor(self):
        curve = [
            {"threshold": 0.2, "tp": 40, "fp": 60},
            {"threshold": 0.3, "tp": 35, "fp": 20},
            {"threshold": 0.4, "tp": 20, "fp": 5},
        ]
        assert optimal_threshold(curve, 1000, 100, 0.3) == 0.3

    def test_em_empate_fica_com_o_threshold_maior(self):
        curve = [{"threshold": 0.2, "tp": 10, "fp": 0}, {"threshold": 0.5, "tp": 10, "fp": 0}]
        assert optimal_threshold(curve, 1000, 100, 0.3) == 0.5

    def test_threshold_teorico(self):
        assert theoretical_threshold(1000, 100, 0.3) == pytest.approx(1 / 3)
        assert theoretical_threshold(1000, 0, 0.3) == 0.0
        assert theoretical_threshold(100, 1000, 0.3) == 1.0
        assert theoretical_threshold(0, 100, 0.3) == 1.0

    def test_valor_esperado_de_contatar_zera_no_threshold_teorico(self):
        t = theoretical_threshold(1000, 100, 0.3)
        assert expected_contact_value(t, 1000, 100, 0.3) == pytest.approx(0.0, abs=1e-9)
        assert expected_contact_value(0.8, 1000, 100, 0.3) == pytest.approx(140.0)
        assert expected_contact_value(0.1, 1000, 100, 0.3) < 0

    def test_threshold_do_treino_fica_perto_do_teorico(self):
        """Com probabilidades calibradas, o ponto ótimo nos dados deve cair
        perto de custo / (taxa de sucesso × LTV)."""
        import json

        evaluation = json.loads((MODELS_DIR / "evaluation.json").read_text(encoding="utf-8"))
        assert abs(evaluation["optimal_threshold"] - evaluation["theoretical_threshold"]) <= 0.05


class TestFaixasDeRisco:
    def test_cortes_derivados_do_threshold(self):
        assert risk_cuts(0.33) == {"baixo_max": 0.33, "medio_max": 0.5}
        assert risk_cuts(0.6) == {"baixo_max": 0.6, "medio_max": 0.6}

    @pytest.mark.parametrize(
        "p,expected",
        [(0.0, "Baixo"), (0.329, "Baixo"), (0.33, "Médio"), (0.49, "Médio"), (0.5, "Alto")],
    )
    def test_fronteiras(self, p, expected):
        assert risk_level(p, risk_cuts(0.33)) == expected


class TestCenarios:
    def test_todos_os_cenarios_sao_clientes_consistentes(self):
        for payload in (make_customer(), loyal_customer()):
            for label, scenario in whatif_scenarios(payload):
                assert consistency_errors(scenario) == [], label

    def test_cada_cenario_muda_alguma_coisa(self):
        payload = make_customer()
        for _, scenario in whatif_scenarios(payload):
            assert scenario != payload

    def test_sem_internet_nao_oferece_servicos_de_internet(self):
        labels = [label for label, _ in whatif_scenarios(loyal_customer())]
        assert not any("suporte" in label or "segurança" in label for label in labels)

    def test_quem_ja_tem_contrato_bienal_nao_recebe_esse_cenario(self):
        labels = [label for label, _ in whatif_scenarios(loyal_customer())]
        assert "Migrar para contrato bienal" not in labels
        assert "Migrar para contrato anual" in labels


@pytest.fixture(scope="module")
def profile():
    typical = pd.read_csv(DATA_DIR / "exemplo_lote.csv")
    return build_reference_profile(typical, CATEGORICAL_COLS + ["SeniorCitizen"], NUMERIC_COLS)


class TestDrift:
    def test_proporcoes_do_perfil_somam_1(self, profile):
        for shares in profile["categorical"].values():
            assert sum(shares.values()) == pytest.approx(1.0)
        for ref in profile["numeric"].values():
            assert sum(ref["shares"]) == pytest.approx(1.0)

    def test_mesma_base_nao_tem_drift(self, profile):
        typical = pd.read_csv(DATA_DIR / "exemplo_lote.csv")
        assert all(row["psi"] < 1e-6 for row in drift_report(typical, profile))

    def test_base_de_clientes_novos_acusa_mudanca_forte_no_tempo_de_casa(self, profile):
        newcomers = pd.read_csv(DATA_DIR / "exemplo_lote_novos.csv")
        report = {row["variavel"]: row for row in drift_report(newcomers, profile)}
        assert report["tenure"]["psi"] > PSI_SHIFT
        assert report["tenure"]["status"] == "Mudança forte"
        # Ordenado do maior para o menor PSI.
        values = [row["psi"] for row in drift_report(newcomers, profile)]
        assert values == sorted(values, reverse=True)

    def test_perfil_de_referencia_do_treino_existe_e_cobre_as_variaveis(self):
        import json

        profile = json.loads((MODELS_DIR / "reference_profile.json").read_text(encoding="utf-8"))
        assert set(profile["numeric"]) == set(NUMERIC_COLS)
        assert set(CATEGORICAL_COLS) <= set(profile["categorical"])

    @pytest.mark.parametrize(
        "value,status",
        [
            (0.0, "Estável"),
            (PSI_STABLE - 1e-9, "Estável"),
            (PSI_STABLE, "Atenção"),
            (PSI_SHIFT, "Mudança forte"),
        ],
    )
    def test_psi_status(self, value, status):
        assert psi_status(value) == status

    def test_categoria_nova_nao_quebra_o_psi(self, profile):
        batch = pd.read_csv(DATA_DIR / "exemplo_lote.csv")
        batch.loc[:10, "Contract"] = "Contrato inexistente"
        report = drift_report(batch, profile)
        assert all(np.isfinite(row["psi"]) for row in report)


class TestFormatacao:
    def test_numeros_no_padrao_brasileiro(self):
        assert fmt_int(1234567) == "1.234.567"
        assert fmt_num(1234.5) == "1.234,50"
        assert fmt_pct(0.2657) == "26,6%"
        assert fmt_brl(21890.5) == "R$ 21.890"
        assert fmt_brl(85.5, cents=True) == "R$ 85,50"
        assert fmt_brl(-1500) == "−R$ 1.500"

    def test_pontos_percentuais_com_sinal(self):
        assert fmt_pp(0.123) == "+12,3 pp"
        assert fmt_pp(-0.05) == "−5,0 pp"
        assert fmt_pp(0.0) == "0,0 pp"
