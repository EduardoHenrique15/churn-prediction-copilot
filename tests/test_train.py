"""Testes de src/train.py — métricas, seleção de modelo e artefatos gerados.

Não treinam o modelo de verdade (isso leva ~30 s e precisa do dataset
bruto, que não vai para o repositório). Testam as peças com dados pequenos
e conferem a coerência dos artefatos versionados em models/.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.business import risk_cuts
from src.train import (
    COMPLEXITY_ORDER,
    EXCLUDED_FEATURES,
    corrected_standard_error,
    expected_calibration_error,
    linear_coefficients,
    load_raw_data,
    operational_metrics,
    reliability_bins,
    select_model,
    sweep_thresholds,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TestMetricas:
    def test_ece_zero_para_previsoes_perfeitamente_calibradas(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=200_000)
        y = rng.uniform(size=p.size) < p
        assert expected_calibration_error(y, p) < 0.01

    def test_ece_detecta_probabilidades_infladas(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=50_000)
        y = rng.uniform(size=p.size) < p
        assert expected_calibration_error(y, np.clip(p + 0.15, 0, 1)) > 0.1

    def test_reliability_bins_cobrem_todos_os_clientes(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(size=1000)
        y = rng.uniform(size=p.size) < p
        bins = reliability_bins(y, p)
        assert sum(b["n"] for b in bins) == 1000
        assert all(b["low"] <= b["mean_predicted"] <= b["high"] for b in bins)

    def test_sweep_thresholds_monta_a_matriz_de_confusao_em_101_pontos(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
        curve = sweep_thresholds(y, p)
        assert len(curve) == 101
        assert all(pt["tp"] + pt["fn"] == 3 and pt["fp"] + pt["tn"] == 2 for pt in curve)
        contacts = [pt["tp"] + pt["fp"] for pt in curve]
        assert contacts == sorted(contacts, reverse=True)
        assert curve[0]["threshold"] == 0.0 and curve[-1]["threshold"] == 1.0

    def test_operational_metrics_no_threshold_escolhido(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0.9, 0.2, 0.7, 0.1])
        m = operational_metrics(y, p, 0.5, 1000, 100, 0.3)
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["contact_rate"] == 0.5
        # 1 TP × 300 − 2 contatos × 100 = 100 em 4 clientes → 25.000 por 1.000
        assert m["net_value_per_1000"] == pytest.approx(25_000)


class TestSelecao:
    def test_erro_padrao_corrigido_e_maior_que_o_ingenuo(self):
        scores = np.array([0.65, 0.66, 0.64, 0.67, 0.66, 0.65, 0.63, 0.68, 0.66, 0.65])
        naive = scores.std(ddof=1) / np.sqrt(len(scores))
        assert corrected_standard_error(scores) > naive

    def test_regra_de_1_erro_padrao_escolhe_o_mais_simples_entre_os_empatados(self):
        cv = {
            "dummy_baseline": {"ap_mean": 0.27, "ap_se": 0.0},
            "logistic_regression": {"ap_mean": 0.660, "ap_se": 0.01},
            "random_forest": {"ap_mean": 0.666, "ap_se": 0.01},
            "hist_gradient_boosting": {"ap_mean": 0.668, "ap_se": 0.01},
        }
        chosen, info = select_model(cv)
        assert chosen == "logistic_regression"
        assert info["best_mean"] == "hist_gradient_boosting"
        assert info["tied"] == COMPLEXITY_ORDER

    def test_sem_empate_fica_o_melhor(self):
        cv = {
            "logistic_regression": {"ap_mean": 0.60, "ap_se": 0.005},
            "random_forest": {"ap_mean": 0.61, "ap_se": 0.005},
            "hist_gradient_boosting": {"ap_mean": 0.68, "ap_se": 0.005},
        }
        assert select_model(cv)[0] == "hist_gradient_boosting"

    def test_baseline_nunca_e_escolhido(self):
        cv = {
            "dummy_baseline": {"ap_mean": 0.99, "ap_se": 0.0},
            "logistic_regression": {"ap_mean": 0.6, "ap_se": 0.01},
        }
        assert select_model(cv)[0] == "logistic_regression"


class TestDados:
    def test_load_raw_data_converte_total_gasto_antes_de_descartar_vazios(self, tmp_path):
        csv = tmp_path / "mini.csv"
        csv.write_text(
            "customerID,tenure,TotalCharges,Churn\na,1,29.85,No\nb,0, ,Yes\nc,2,108.15,Yes\n",
            encoding="utf-8",
        )
        df = load_raw_data(str(csv))
        assert len(df) == 2
        assert "customerID" not in df.columns
        assert df["Churn"].tolist() == [0, 1]
        assert load_raw_data(str(csv), keep_id=True)["customerID"].tolist() == ["a", "c"]

    def test_coeficientes_so_para_modelo_linear(self):
        X = pd.DataFrame({"a": [0, 1, 0, 1], "b": [1, 1, 0, 0]})
        forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, [0, 1, 0, 1])
        assert linear_coefficients(forest, list(X.columns)) is None


@pytest.fixture(scope="module")
def evaluation():
    return json.loads((MODELS_DIR / "evaluation.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metrics():
    return json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))


class TestArtefatos:
    def test_modelo_nao_usa_as_variaveis_redundantes(self):
        columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
        assert not set(EXCLUDED_FEATURES) & set(columns)

    def test_faixas_de_risco_derivam_do_threshold(self, evaluation):
        assert evaluation["risk_level_cuts"] == risk_cuts(evaluation["optimal_threshold"])

    def test_metricas_e_avaliacao_descrevem_o_mesmo_treino(self, evaluation, metrics):
        for key in ("schema_version", "model_selected", "trained_at", "model_version"):
            assert evaluation[key] == metrics[key]
        assert metrics["operational"]["threshold"] == evaluation["optimal_threshold"]

    def test_intervalos_de_confianca_contem_o_valor_do_teste(self, metrics):
        for key in ("average_precision", "roc_auc", "brier"):
            low, high = metrics["ci95"][key]
            assert low <= metrics["test_metrics"][key] <= high

    def test_modelo_supera_o_baseline(self, metrics):
        cv = metrics["cv_comparison"]
        assert (
            cv[metrics["model_selected"]]["cv_ap_mean"] > cv["dummy_baseline"]["cv_ap_mean"] + 0.2
        )

    def test_curvas_tem_101_pontos(self, evaluation):
        assert len(evaluation["curves"]["oof"]) == 101
        assert len(evaluation["curves"]["test"]) == 101

    def test_previsoes_de_teste_batem_com_o_tamanho_do_teste(self, metrics):
        preds = pd.read_csv(MODELS_DIR / "test_predictions.csv")
        assert len(preds) == metrics["n_test"]
        assert preds["y_proba"].between(0, 1).all()