"""Testes de src/tools.py — as ferramentas do agente.

As ferramentas são StructuredTools do LangChain: chamamos via .invoke(),
como o agente faz. Com argumentos soltos, `invoke` devolve só o conteúdo;
com uma tool call completa, devolve o ToolMessage (conteúdo + artefato).
"""

from unittest.mock import MagicMock, patch

import pytest

import src.tools as tools
from src.client import ChurnClient
from src.tools import predict_churn, search_churn_knowledge
from tests.conftest import make_customer


@pytest.fixture(autouse=True)
def _cliente_local():
    previous = tools._client
    tools.set_client(ChurnClient(api_url=None))
    yield
    tools._client = previous


def test_previsao_traz_resultado_e_principais_fatores_em_portugues():
    result = predict_churn.invoke(make_customer())
    assert set(result) >= {
        "churn_prediction",
        "churn_probability",
        "risk_level",
        "principais_fatores",
    }
    first = result["principais_fatores"][0]
    assert first["efeito"] in {"aumenta o risco", "reduz o risco"}
    assert ":" in first["fator"]
    # Campos agrupados: "Contrato: Mensal", não "Contrato: não é bienal".
    assert all("não é" not in f["fator"] for f in result["principais_fatores"])


def test_cliente_inconsistente_devolve_erro_estruturado_para_o_modelo_corrigir():
    result = predict_churn.invoke(make_customer(TechSupport="No internet service"))
    assert result["error"].startswith("Valores inválidos")
    assert result["detail"]


def test_falha_inesperada_vira_erro_legivel_e_nao_derruba_a_conversa():
    broken = MagicMock()
    broken.predict.side_effect = RuntimeError("modelo corrompido")
    tools.set_client(broken)
    result = predict_churn.invoke(make_customer())
    assert "error" in result


def test_explicacao_indisponivel_nao_impede_a_previsao():
    client = ChurnClient(api_url=None)
    client.explain = MagicMock(side_effect=RuntimeError("sem SHAP"))
    tools.set_client(client)
    result = predict_churn.invoke(make_customer())
    assert "churn_probability" in result
    assert "principais_fatores" not in result


def test_tool_call_completa_devolve_toolmessage_com_a_fonte_no_artefato():
    message = predict_churn.invoke(
        {"name": "predict_churn", "args": make_customer(), "id": "call_1", "type": "tool_call"}
    )
    assert message.artifact["fonte"] == "local"
    assert "churn_probability" in message.content


def test_busca_sem_base_construida_orienta_em_vez_de_quebrar():
    with patch("src.tools.os.path.isdir", return_value=False):
        message = search_churn_knowledge.invoke(
            {
                "name": "search_churn_knowledge",
                "args": {"query": "churn"},
                "id": "1",
                "type": "tool_call",
            }
        )
    assert "build_knowledge_base" in message.content
    assert message.artifact == {"sources": []}


def test_busca_devolve_trechos_e_fontes_sem_duplicar():
    class Doc:
        def __init__(self, source):
            self.page_content = f"trecho de {source}"
            self.metadata = {"source": f"/abs/{source}"}

    retriever = MagicMock()
    retriever.invoke.return_value = [Doc("a.md"), Doc("b.md"), Doc("a.md")]
    with (
        patch("src.tools.os.path.isdir", return_value=True),
        patch("src.tools._get_retriever", return_value=retriever),
    ):
        message = search_churn_knowledge.invoke(
            {
                "name": "search_churn_knowledge",
                "args": {"query": "x"},
                "id": "1",
                "type": "tool_call",
            }
        )
    assert message.artifact == {"sources": ["a.md", "b.md"]}
    assert "trecho de b.md" in message.content
