"""Eval de roteamento do agente — roda contra o Gemini DE VERDADE.

⚠️ Gasta cota: 6 requisições por execução (uma por pergunta). A cota
gratuita é de ~20 por dia. Fica fora do `pytest` padrão (ver addopts em
pyproject.toml); rode só quando quiser, com:

    pytest -m integration

Mede só a PRIMEIRA decisão do modelo (`route`): qual ferramenta ele chama
para cada tipo de pergunta — prever um cliente, consultar a base de
conhecimento ou recusar algo fora do escopo. O texto da resposta muda a
cada versão do Gemini e não é o que este eval quer travar.
"""

import os

import pytest

from src.agent import route

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY não configurada — eval do agente pulado",
    ),
]

CASES = [
    (
        "Cliente mulher, não idosa, sem cônjuge, sem dependentes, 3 meses de casa, com "
        "telefone, sem múltiplas linhas, internet fibra óptica, sem segurança online, sem "
        "backup online, sem proteção de aparelho, sem suporte técnico, com streaming de TV "
        "e de filmes, contrato mensal, fatura digital, pagamento por cheque eletrônico, "
        "mensalidade de 95 reais, total gasto de 285 reais. Qual o risco de cancelamento?",
        {"predict_churn"},
    ),
    (
        "Tenho um cliente homem, idoso, com cônjuge e dependentes, 60 meses de casa, "
        "contrato de dois anos, com telefone e múltiplas linhas, internet DSL, com "
        "segurança online, backup online, proteção de aparelho e suporte técnico, sem "
        "streaming, fatura em papel, pagamento por transferência bancária automática, "
        "mensalidade de 45 reais, total gasto de 2700 reais. Ele tem risco alto de churn?",
        {"predict_churn"},
    ),
    ("O que é churn?", {"search_churn_knowledge"}),
    ("Quais são os principais fatores de risco de churn em telecom?", {"search_churn_knowledge"}),
    ("Como devo interpretar a faixa de risco Médio?", {"search_churn_knowledge"}),
    ("Qual é a capital da França?", set()),
]


@pytest.mark.parametrize("prompt,expected", CASES)
def test_agente_escolhe_a_ferramenta_certa(prompt, expected):
    called = set(route(prompt))
    if expected:
        assert expected <= called, f"Esperava {expected}, veio {called} para: {prompt!r}"
    else:
        assert not called, f"Pergunta fora do escopo não deveria chamar ferramenta: {called}"
