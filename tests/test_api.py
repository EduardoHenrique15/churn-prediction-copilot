"""Testes do contrato HTTP da API (src/api.py).

O TestClient chama a aplicação ASGI em processo, sem subir servidor. Os
artefatos em models/ já vêm versionados, então nada aqui depende de treinar.
"""

import math

import pytest
from fastapi.testclient import TestClient

import src.api as api_module
from src.api import DECISION_THRESHOLD, MAX_BATCH_SIZE, RISK_LEVEL_CUTS, app, expected_columns
from tests.conftest import NO_INTERNET_SERVICES, loyal_customer, make_customer

client = TestClient(app)


def _proba(payload: dict) -> float:
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["churn_probability"]


@pytest.fixture(autouse=True)
def _sem_limite_de_taxa(monkeypatch):
    """Os testes disparam dezenas de requisições do mesmo "IP"."""
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 0)
    api_module._hits.clear()


class TestHealth:
    def test_responde_online_com_metadados_do_modelo(self):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["decision_threshold"] == DECISION_THRESHOLD
        assert body["risk_level_cuts"] == RISK_LEVEL_CUTS
        assert body["model_features"] == len(expected_columns)
        for field in ("api_version", "model_selected", "trained_at"):
            assert body[field]

    def test_threshold_e_faixas_vem_do_treino_nao_de_constantes(self):
        """0,5 / 0,35 / 0,65 eram números fixos no código antigo."""
        assert DECISION_THRESHOLD != 0.5
        assert 0 < RISK_LEVEL_CUTS["baixo_max"] <= RISK_LEVEL_CUTS["medio_max"] < 1

    def test_toda_resposta_traz_o_tempo_de_processamento(self):
        response = client.post("/predict", json=make_customer())
        assert float(response.headers["X-Response-Time-ms"]) >= 0

    def test_preflight_de_cors_para_consumo_pelo_navegador(self):
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://exemplo.dev",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestPredict:
    def test_contrato_da_resposta_continua_o_mesmo(self):
        response = client.post("/predict", json=make_customer())
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"churn_prediction", "churn_probability", "risk_level"}
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_level"] in {"Baixo", "Médio", "Alto"}

    @pytest.mark.parametrize(
        "field,value",
        [
            ("Contract", "Two year"),
            ("Contract", "One year"),
            ("InternetService", "DSL"),
            ("PaymentMethod", "Credit card (automatic)"),
            ("TechSupport", "Yes"),
        ],
    )
    def test_mudar_uma_categoria_muda_a_previsao(self, field, value):
        """Regressão do bug de encoding: antes, estas mudanças devolviam
        exatamente a mesma probabilidade (0,758) para qualquer valor."""
        assert _proba(make_customer(**{field: value})) != _proba(make_customer())

    def test_contrato_longo_reduz_o_risco(self):
        mensal = _proba(make_customer())
        assert _proba(make_customer(Contract="One year")) < mensal
        assert _proba(make_customer(Contract="Two year")) < _proba(
            make_customer(Contract="One year")
        )

    def test_decisao_e_faixa_seguem_o_threshold_do_treino(self):
        for payload in (make_customer(), loyal_customer()):
            body = client.post("/predict", json=payload).json()
            p = body["churn_probability"]
            assert body["churn_prediction"] == (p >= DECISION_THRESHOLD)
            expected = (
                "Alto"
                if p >= RISK_LEVEL_CUTS["medio_max"]
                else "Médio"
                if p >= RISK_LEVEL_CUTS["baixo_max"]
                else "Baixo"
            )
            assert body["risk_level"] == expected

    def test_perfil_fiel_tem_risco_baixo(self):
        body = client.post("/predict", json=loyal_customer()).json()
        assert body["risk_level"] == "Baixo"
        assert body["churn_prediction"] is False

    @pytest.mark.parametrize(
        "field,value",
        [
            ("gender", "Banana"),
            ("tenure", -5),
            ("MonthlyCharges", -10.0),
            ("SeniorCitizen", 2),
            ("Contract", "Lifetime"),
            ("PaymentMethod", "Dinheiro"),
        ],
    )
    def test_rejeita_valores_fora_do_dominio(self, field, value):
        assert client.post("/predict", json=make_customer(**{field: value})).status_code == 422

    def test_rejeita_campo_obrigatorio_ausente(self):
        payload = make_customer()
        del payload["tenure"]
        assert client.post("/predict", json=payload).status_code == 422

    @pytest.mark.parametrize(
        "overrides",
        [
            {**NO_INTERNET_SERVICES, "TechSupport": "Yes"},
            {"OnlineSecurity": "No internet service"},
            {"PhoneService": "No", "MultipleLines": "Yes"},
            {"MultipleLines": "No phone service"},
            {"TotalCharges": 100.0},
        ],
    )
    def test_rejeita_combinacoes_impossiveis_com_mensagem_em_portugues(self, overrides):
        response = client.post("/predict", json=make_customer(**overrides))
        assert response.status_code == 422
        assert "Cliente" in str(response.json()) or "TotalCharges" in str(response.json())


class TestBatch:
    def test_cliente_no_lote_tem_o_mesmo_resultado_que_sozinho(self):
        individual = client.post("/predict", json=make_customer()).json()
        batch = client.post("/predict/batch", json={"customers": [make_customer()]}).json()
        assert batch["predictions"][0] == individual

    def test_resultado_de_um_cliente_nao_depende_dos_vizinhos(self):
        """Regressão: a previsão do mesmo cliente variava de 0,758 para
        0,034 conforme o resto do lote."""
        alvo = make_customer()
        vizinhos = [
            loyal_customer(),
            make_customer(Contract="One year", tenure=30, TotalCharges=2565),
        ]
        sozinho = client.post("/predict/batch", json={"customers": [alvo]}).json()
        junto = client.post("/predict/batch", json={"customers": [*vizinhos, alvo]}).json()
        assert junto["predictions"][-1] == sozinho["predictions"][0]

    def test_preserva_a_ordem_dos_clientes(self):
        customers = [make_customer(), loyal_customer(), make_customer()]
        predictions = client.post("/predict/batch", json={"customers": customers}).json()[
            "predictions"
        ]
        assert len(predictions) == 3
        assert predictions[0] == predictions[2]
        assert predictions[1]["churn_probability"] < predictions[0]["churn_probability"]

    def test_rejeita_lote_vazio(self):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422

    def test_rejeita_lote_acima_do_teto(self):
        customers = [make_customer()] * (MAX_BATCH_SIZE + 1)
        assert client.post("/predict/batch", json={"customers": customers}).status_code == 422

    def test_rejeita_o_lote_se_qualquer_cliente_for_invalido(self):
        payload = {"customers": [make_customer(), make_customer(gender="Banana")]}
        assert client.post("/predict/batch", json=payload).status_code == 422


class TestExplain:
    def test_contribuicoes_reproduzem_a_probabilidade_do_predict(self):
        """base_value + soma das contribuições = saída do modelo no espaço
        `link` — é essa aditividade que torna a explicação honesta."""
        payload = make_customer()
        body = client.post("/explain", json=payload).json()
        assert body["churn_probability"] == _proba(payload)
        total = body["base_value"] + sum(c["shap_value"] for c in body["contributions"])
        reconstructed = 1 / (1 + math.exp(-total)) if body["link"] == "logit" else total
        assert reconstructed == pytest.approx(body["churn_probability"], abs=1e-3)

    def test_ordena_por_magnitude_decrescente(self):
        body = client.post("/explain", json=make_customer()).json()
        magnitudes = [abs(c["shap_value"]) for c in body["contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_cobre_todas_as_colunas_do_modelo(self):
        body = client.post("/explain", json=make_customer()).json()
        assert {c["feature"] for c in body["contributions"]} == set(expected_columns)

    def test_rejeita_payload_invalido_igual_ao_predict(self):
        assert client.post("/explain", json=make_customer(gender="Banana")).status_code == 422

    def test_modelo_linear_explica_mesmo_com_shap_desligado(self, monkeypatch):
        """A explicação da regressão logística é exata e não usa SHAP."""
        monkeypatch.setattr(api_module, "ENABLE_SHAP_EXPLAIN", False)
        if not api_module.predictor.is_linear:
            pytest.skip("o modelo publicado não é linear")
        assert client.post("/explain", json=make_customer()).status_code == 200

    def test_modelo_de_arvore_com_shap_desligado_responde_503(self, monkeypatch):
        class TreePredictor:
            is_linear = False

        monkeypatch.setattr(api_module, "ENABLE_SHAP_EXPLAIN", False)
        monkeypatch.setattr(api_module, "predictor", TreePredictor())
        assert client.post("/explain", json=make_customer()).status_code == 503


class TestRateLimit:
    def test_bloqueia_com_429_acima_do_limite_por_minuto(self, monkeypatch):
        monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 2)
        api_module._hits.clear()
        statuses = [client.post("/predict", json=make_customer()).status_code for _ in range(3)]
        assert statuses == [200, 200, 429]

    def test_health_nao_conta_no_limite(self, monkeypatch):
        monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 1)
        api_module._hits.clear()
        assert all(client.get("/health").status_code == 200 for _ in range(3))

    def test_ip_real_vem_do_x_forwarded_for(self, monkeypatch):
        monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 1)
        api_module._hits.clear()
        first = client.post(
            "/predict", json=make_customer(), headers={"X-Forwarded-For": "1.1.1.1"}
        )
        other = client.post(
            "/predict", json=make_customer(), headers={"X-Forwarded-For": "2.2.2.2"}
        )
        assert (first.status_code, other.status_code) == (200, 200)


def test_rate_limit_nao_e_driblado_trocando_o_x_forwarded_for(monkeypatch):
    """O cliente controla o começo do X-Forwarded-For; só o último endereço
    (anexado pelo proxy) identifica quem está chamando."""
    monkeypatch.setattr(api_module, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(api_module, "_hits", api_module.defaultdict(api_module.deque))
    codes = [
        client.post(
            "/predict",
            json=make_customer(),
            headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"},
        ).status_code
        for i in range(5)
    ]
    assert codes.count(429) == 2
