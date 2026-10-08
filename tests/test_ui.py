"""Testes da interface com o AppTest do Streamlit — sem navegador e sem rede.

A API fica desligada (CHURN_API_URL vazio): a interface calcula as previsões
localmente, com o mesmo modelo. O assistente usa um modelo de linguagem
falso — nenhuma chamada ao Gemini.
"""

import os

os.environ["CHURN_API_URL"] = ""  # antes de importar a interface: só cálculo local

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessageChunk
from streamlit.testing.v1 import AppTest

from tests.conftest import make_customer

TIMEOUT = 60
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def page(name: str) -> AppTest:
    return AppTest.from_string(
        f"from src.ui.pages import {name}\n{name}.render()", default_timeout=TIMEOUT
    )


def html(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


@pytest.fixture(autouse=True)
def _sem_chave_do_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_css_do_tema_nao_tem_sinal_de_menor():
    """Um "<p>" num comentário do CSS bastou para o sanitizador do st.html
    descartar o <style> inteiro — a interface perdia o tema sem nenhum erro."""
    from src.ui.theme import STYLES

    css = STYLES.strip().removeprefix("<style>").removesuffix("</style>")
    assert "<" not in css


def test_tabela_leva_o_nome_da_coluna_em_cada_celula():
    """No celular a tabela vira cartões e cada valor mostra o nome da coluna
    via data-label — sem isso só a 1ª coluna aparecia na tela."""
    from src.ui.components import table

    markup = table(["Grupo", "Clientes"], [["Idosos", "232"], ["Não idosos", "1.175"]], highlight=1)
    assert 'data-label="Clientes">232</td>' in markup
    assert '<tr class="cr-hl"><td data-label="Grupo">Não idosos' in markup
    assert "cr-table--stack" in markup


def test_app_completo_abre_na_visao_geral():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Churn Radar" in html(at)


@pytest.mark.parametrize(
    "name", ["inicio", "cliente", "carteira", "estrategia", "modelo", "assistente"]
)
def test_paginas_abrem_sem_erro(name):
    at = page(name).run()
    assert not at.exception, at.exception


class TestCliente:
    def test_mostra_veredito_explicacao_e_cenarios(self):
        text = html(page("cliente").run())
        for piece in ("de chance de cancelar", "O que mais pesou", "O que mudaria a previsão"):
            assert piece in text

    def test_perfil_de_baixo_risco_nao_recomenda_contato(self):
        at = page("cliente").run()
        at.pills(key="f_profile").set_value("Baixo risco").run()
        assert not at.exception
        assert "Não priorizar agora" in html(at)

    def test_cliente_real_mostra_o_desfecho_verdadeiro(self):
        at = page("cliente").run()
        at.pills(key="f_profile").set_value("Cliente real").run()
        assert "do conjunto de teste" in html(at)

    def test_combinacao_impossivel_e_bloqueada_com_explicacao(self):
        at = page("cliente").run()
        at.segmented_control(key="f_internet").set_value("No").run()
        at.toggle(key="f_phone").set_value(False).run()
        assert "Combinação impossível" in html(at)

    def test_sem_internet_limpa_os_servicos_de_internet(self):
        at = page("cliente").run()
        at.segmented_control(key="f_internet").set_value("No").run()
        assert at.session_state["f_services"] == []


class TestCarteira:
    def test_exemplo_tipico_e_estavel_e_confere_com_o_real(self):
        text = html(page("carteira").run())
        assert "prontas para prever" in text
        assert "O modelo contra o que aconteceu de verdade" in text
        assert "nenhuma variável com mudança forte" in text

    def test_exemplo_de_clientes_novos_acusa_mudanca_forte(self):
        at = page("carteira").run()
        at.segmented_control(key="b_source").set_value("novos").run()
        assert "Mudança forte" in html(at)

    def test_upload_com_linhas_invalidas_preve_as_validas_e_lista_as_demais(self):
        rows = [make_customer(), make_customer(Contract="Vitalício"), make_customer(tenure=-1)]
        csv = (
            ",".join(rows[0]) + "\n" + "\n".join(",".join(str(v) for v in r.values()) for r in rows)
        )
        at = page("carteira").run()
        at.segmented_control(key="b_source").set_value("upload").run()
        at.file_uploader(key="b_file").upload("clientes.csv", csv.encode("utf-8"), "text/csv").run()
        assert not at.exception
        text = html(at)
        assert "<b>3</b> linhas lidas" in text
        assert "<b>2</b> com problema" in text

    def test_arquivo_sem_colunas_obrigatorias_explica_o_que_falta(self):
        at = page("carteira").run()
        at.segmented_control(key="b_source").set_value("upload").run()
        at.file_uploader(key="b_file").upload("ruim.csv", b"a,b\n1,2\n", "text/csv").run()
        assert "Faltam colunas obrigatórias" in html(at)


class TestHipotesesCompartilhadas:
    """As hipóteses de custo ajustadas na Estratégia valem na Carteira."""

    @staticmethod
    def _contatar(at: AppTest) -> int:
        text = html(at)
        start = text.index("A contatar")
        return int(text[start:].split('cr-kpi-value">', 1)[1].split("<", 1)[0].replace(".", ""))

    def test_sem_ajuste_a_carteira_decide_igual_a_api(self):
        """Com as hipóteses de referência, a fila é exatamente o
        churn_prediction que a API devolve para cada cliente."""
        from src.ui.data import example_batch, get_client

        at = page("carteira").run()
        assert "Hipóteses de referência" in html(at)
        records = example_batch("tipica").drop(columns=["customerID", "Churn"])
        api = get_client().predict(records.to_dict("records")).data
        assert self._contatar(at) == sum(p["churn_prediction"] for p in api)

    def test_oferta_mais_cara_encolhe_a_fila_de_contato(self):
        reference = self._contatar(page("carteira").run())
        at = page("carteira")
        at.session_state["s_offer"] = 300
        at.run()
        assert not at.exception
        assert "Hipóteses ajustadas na página Estratégia" in html(at)
        assert self._contatar(at) < reference

    def test_oferta_mais_cara_que_o_retorno_explica_que_ninguem_compensa(self):
        """R$ 300 de oferta × 30% × R$ 1.000: o retorno máximo (R$ 300) não
        cobre o custo — o corte vai a 100% e a nota diz isso com palavras."""
        at = page("carteira")
        at.session_state["s_offer"] = 300
        at.run()
        assert self._contatar(at) == 0
        assert "nenhum contato compensa" in html(at)

    def test_estrategia_guarda_o_ajuste_e_permite_voltar_a_referencia(self):
        at = page("estrategia").run()
        assert not [b for b in at.button if "referência" in b.label]
        at.slider(key="s_offer").set_value(300).run()
        assert at.session_state["s_offer"] == 300
        [reset] = [b for b in at.button if "referência" in b.label]
        reset.click().run()
        assert at.session_state["s_offer"] == 100


def test_estrategia_mostra_corte_recomendado():
    text = html(page("estrategia").run())
    assert "Corte recomendado" in text
    assert "Formas de montar a campanha" in text


def test_modelo_mostra_escolha_calibracao_e_limites():
    text = html(page("modelo").run())
    for piece in (
        "Por que uma regressão logística",
        "Dá para ler a probabilidade",
        "Não serve para",
    ):
        assert piece in text


class FakeLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)

    def stream(self, messages):
        yield from self.rounds.pop(0)


class TestAssistente:
    def test_sem_chave_fica_em_modo_demonstracao(self):
        at = page("assistente").run()
        assert "modo demonstração" in html(at)
        assert at.chat_input[0].proto.disabled

    def test_pergunta_ao_vivo_com_modelo_falso_mostra_ferramenta_e_resposta(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "chave-de-teste")
        tool_call = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "predict_churn",
                    "args": json.dumps(make_customer()),
                    "id": "c1",
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        )
        fake = FakeLLM(
            [[tool_call], [AIMessageChunk(content="O risco é **alto**: R$ 124 de valor esperado.")]]
        )
        monkeypatch.setattr("src.agent._get_llm", lambda: fake)

        at = page("assistente").run()
        at.chat_input[0].set_value("Qual o risco deste cliente?").run()
        assert not at.exception
        text = html(at)
        assert "previsão do modelo" in text
        assert "R\\$ 124" in text  # cifrão escapado: não vira fórmula no Markdown


def test_assistente_mostra_resposta_gravada_com_data_e_modelo(tmp_path, monkeypatch):
    recording = {
        "gravado_em": "2026-09-26T12:00:00+00:00",
        "modelo": "gemini-teste",
        "respostas": {
            "prever": {
                "titulo": "Prever um cliente",
                "pergunta": "Qual o risco?",
                "resposta": "Risco de **77%**, vale contatar (R$ 132 de valor esperado).",
                "ferramentas": [{"tool": "predict_churn", "fonte": "api", "ms": 150}],
                "estatisticas": {"model_calls": 2, "latency_ms": 2100},
            }
        },
    }
    path = tmp_path / "agent_demo.json"
    path.write_text(json.dumps(recording), encoding="utf-8")
    monkeypatch.setattr("src.ui.data.DEMO_PATH", path)

    at = page("assistente").run()
    text = html(at)
    assert "Resposta gravada em 26/09/2026 com gemini-teste" in text
    assert "previsão do modelo" in text
    # Perguntas sem gravação ficam com o botão desabilitado.
    assert at.button(key="demo_fatores").disabled
    assert not at.button(key="demo_prever").disabled
