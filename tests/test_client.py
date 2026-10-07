"""Testes de src/client.py — API com plano B local, sem rede de verdade."""

import time

import pytest
import requests

import src.client as client_module
from src.client import (
    STATE_LOCAL_ONLY,
    STATE_ONLINE,
    ChurnClient,
    InvalidCustomerError,
)
from src.predictor import ChurnPredictor
from src.schema import MAX_BATCH_SIZE
from tests.conftest import loyal_customer, make_customer


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.ok = status_code < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def test_sem_url_calcula_tudo_localmente(local_client):
    answer = local_client.predict([make_customer(), loyal_customer()])
    assert answer.source == "local"
    assert len(answer.data) == 2
    assert local_client.state == STATE_LOCAL_ONLY


def test_resultado_local_e_identico_ao_da_api():
    from fastapi.testclient import TestClient

    from src.api import app

    api = TestClient(app).post("/predict", json=make_customer()).json()
    local = ChurnClient(None).predict([make_customer()]).data[0]
    assert local == api


def test_validacao_local_usa_as_mesmas_regras_da_api(local_client):
    with pytest.raises(InvalidCustomerError) as info:
        local_client.predict([make_customer(), make_customer(OnlineSecurity="No internet service")])
    assert info.value.detail[0]["loc"][:2] == ["customers", 1]


def test_api_fora_do_ar_cai_no_plano_b_e_tenta_acordar(monkeypatch):
    calls = []

    def refused(*args, **kwargs):
        calls.append(args[0])
        raise requests.ConnectionError("recusado")

    monkeypatch.setattr(client_module.requests, "post", refused)
    monkeypatch.setattr(client_module.requests, "get", refused)
    client = ChurnClient("http://api.invalida", retry_after=60)

    answer = client.predict([make_customer()])
    assert answer.source == "local"
    deadline = time.time() + 2
    while client.state == "acordando" and time.time() < deadline:
        time.sleep(0.01)
    assert client.state == "offline"
    # Enquanto estiver fora, nem tenta a API de novo (sem pagar o timeout).
    n_calls = len(calls)
    assert client.predict([make_customer()]).source == "local"
    assert len(calls) == n_calls


def test_api_no_ar_responde_pela_api(monkeypatch):
    local = ChurnPredictor()

    def fake_post(url, json, timeout):
        assert url.endswith("/predict/batch")
        return FakeResponse(200, {"predictions": local.predict(json["customers"])})

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = ChurnClient("http://api.exemplo")
    answer = client.predict([make_customer()])
    assert answer.source == "api"
    assert client.state == STATE_ONLINE


def test_422_da_api_nao_cai_no_plano_b(monkeypatch):
    monkeypatch.setattr(
        client_module.requests,
        "post",
        lambda *a, **k: FakeResponse(422, {"detail": [{"msg": "inválido"}]}),
    )
    with pytest.raises(InvalidCustomerError):
        ChurnClient("http://api.exemplo").predict([make_customer()])


def test_lote_grande_e_enviado_em_pedacos(monkeypatch):
    sizes = []

    def fake_post(url, json, timeout):
        sizes.append(len(json["customers"]))
        return FakeResponse(
            200,
            {
                "predictions": [
                    {"churn_prediction": False, "churn_probability": 0.1, "risk_level": "Baixo"}
                ]
                * len(json["customers"])
            },
        )

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    answer = ChurnClient("http://api.exemplo").predict([make_customer()] * (MAX_BATCH_SIZE + 5))
    assert sizes == [MAX_BATCH_SIZE, 5]
    assert len(answer.data) == MAX_BATCH_SIZE + 5


def test_explain_com_503_calcula_localmente_sem_marcar_api_como_fora(monkeypatch):
    monkeypatch.setattr(client_module.requests, "post", lambda *a, **k: FakeResponse(503))
    client = ChurnClient("http://api.exemplo")
    answer = client.explain(make_customer())
    assert answer.source == "local"
    assert client.state != "acordando"
    assert answer.data["contributions"]


def test_check_usa_o_health_e_guarda_a_latencia(monkeypatch):
    monkeypatch.setattr(
        client_module.requests,
        "get",
        lambda *a, **k: FakeResponse(200, {"status": "healthy", "api_version": "2.0.0"}),
    )
    client = ChurnClient("http://api.exemplo")
    assert client.check() == STATE_ONLINE
    status = client.status()
    assert status["health"]["api_version"] == "2.0.0"
    assert status["latency_ms"] is not None
