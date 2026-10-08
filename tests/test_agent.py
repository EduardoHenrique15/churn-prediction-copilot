"""Testes de src/agent.py com um modelo de linguagem FALSO.

Nenhum teste aqui precisa de GEMINI_API_KEY nem de rede: o modelo falso
devolve pedaços roteirizados (texto ou pedidos de ferramenta), do mesmo
jeito que o Gemini faz em streaming. O eval contra o modelo real está em
tests/test_agent_eval.py (marcado como integration, fora do pytest padrão).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import src.agent as agent
import src.tools as tools
from src.agent import (
    AUTH_MESSAGE,
    DEMO_PROMPTS,
    GENERIC_MESSAGE,
    MAX_TOOL_ITERATIONS,
    NETWORK_MESSAGE,
    NOT_CONFIGURED_MESSAGE,
    QUOTA_MESSAGE,
    AgentNotConfiguredError,
    TurnResult,
    cap_history,
    chat,
    friendly_error,
    route,
    run_turn,
)
from src.client import ChurnClient
from tests.conftest import make_customer


class FakeLLM:
    """Cada chamada a stream() consome a próxima rodada roteirizada."""

    def __init__(self, rounds, invoke_result=None):
        self.rounds = list(rounds)
        self.calls: list[list] = []
        self.invoke_result = invoke_result

    def stream(self, messages):
        self.calls.append(list(messages))
        chunks = self.rounds.pop(0)
        if isinstance(chunks, Exception):
            raise chunks
        yield from chunks

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.invoke_result


def text_round(*pieces, usage=None):
    chunks = [AIMessageChunk(content=p) for p in pieces]
    if usage:
        chunks.append(AIMessageChunk(content="", usage_metadata=usage))
    return chunks


def tool_round(name, args, call_id="call_1"):
    """Pedido de ferramenta em DOIS pedaços, como no streaming real: os
    argumentos chegam em partes e precisam ser agregados."""
    raw = json.dumps(args)
    half = len(raw) // 2
    return [
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": name,
                    "args": raw[:half],
                    "id": call_id,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": None,
                    "args": raw[half:],
                    "id": None,
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        ),
    ]


@pytest.fixture
def events():
    return []


@pytest.fixture(autouse=True)
def _cliente_local():
    """A ferramenta de previsão calcula localmente (sem API, sem rede)."""
    previous = tools._client
    tools.set_client(ChurnClient(api_url=None))
    yield
    tools._client = previous


def _run(llm, message, events, history=None):
    with patch("src.agent._get_llm", return_value=llm):
        return run_turn(message, history, on_event=events.append)


class TestRunTurn:
    def test_resposta_sem_ferramenta_usa_uma_unica_chamada(self, events):
        llm = FakeLLM(
            [
                text_round(
                    "Olá",
                    ", tudo bem?",
                    usage={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                )
            ]
        )
        result = _run(llm, "oi", events)

        assert result.answer == "Olá, tudo bem?"
        assert [e["text"] for e in events if e["type"] == "text"] == ["Olá", ", tudo bem?"]
        assert result.stats["model_calls"] == 1
        assert result.stats["input_tokens"] == 12 and result.stats["output_tokens"] == 4
        assert isinstance(result.conversation[-1], AIMessage)

    def test_system_prompt_vai_na_frente_e_historico_e_preservado(self, events):
        history = [HumanMessage(content="pergunta antiga"), AIMessage(content="resposta antiga")]
        llm = FakeLLM([text_round("ok")])
        result = _run(llm, "nova", events, history=history)

        sent = llm.calls[0]
        assert isinstance(sent[0], SystemMessage)
        assert [m.content for m in sent[1:]] == ["pergunta antiga", "resposta antiga", "nova"]
        assert result.conversation[:2] == history

    def test_previsao_executa_a_ferramenta_e_responde_na_rodada_seguinte(self, events):
        llm = FakeLLM([tool_round("predict_churn", make_customer()), text_round("O risco é alto.")])
        result = _run(llm, "qual o risco deste cliente?", events)

        assert result.answer == "O risco é alto."
        assert result.stats["model_calls"] == 2
        assert result.stats["tool_calls"] == 1
        (step,) = result.trace
        assert step["tool"] == "predict_churn"
        assert step["args"] == make_customer()
        assert step["fonte"] == "local"
        assert "churn_probability" in step["result"]
        assert "principais_fatores" in step["result"]
        assert [e["type"] for e in events] == ["reset", "tool", "text"]
        tool_messages = [m for m in result.conversation if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1

    def test_erro_de_validacao_volta_para_o_modelo_corrigir(self, events):
        bad = make_customer(InternetService="No")
        llm = FakeLLM([tool_round("predict_churn", bad), text_round("Faltam dados.")])
        result = _run(llm, "cliente sem internet", events)
        assert "inválidos ou inconsistentes" in result.trace[0]["result"]
        tool_message = next(m for m in result.conversation if isinstance(m, ToolMessage))
        assert "No internet service" in tool_message.content

    def test_rag_registra_os_documentos_consultados(self, events):
        class Doc:
            def __init__(self, content, source):
                self.page_content = content
                self.metadata = {"source": source}

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Doc("Churn é o cancelamento...", "/x/data/knowledge_base/01_o_que_e_churn.md"),
            Doc("Mais contexto...", "/x/data/knowledge_base/01_o_que_e_churn.md"),
            Doc("Fatores...", "/x/data/knowledge_base/02_fatores_de_risco.md"),
        ]
        llm = FakeLLM(
            [
                tool_round("search_churn_knowledge", {"query": "o que é churn"}),
                text_round("Churn é..."),
            ]
        )
        with (
            patch("src.tools._get_retriever", return_value=retriever),
            patch("src.tools._ensure_index", return_value=True),
        ):
            result = _run(llm, "o que é churn?", events)

        assert result.trace[0]["sources"] == ["01_o_que_e_churn.md", "02_fatores_de_risco.md"]

    def test_para_no_limite_de_rodadas_de_ferramenta(self, events):
        rounds = [
            tool_round("search_churn_knowledge", {"query": "x"}, call_id=f"c{i}")
            for i in range(MAX_TOOL_ITERATIONS + 1)
        ]
        with patch("src.tools._ensure_index", return_value=False):
            result = _run(FakeLLM(rounds), "pergunta que nunca conclui", events)

        assert "máximo de consultas" in result.answer
        assert result.stats["model_calls"] == MAX_TOOL_ITERATIONS + 1
        assert result.conversation[-1].content == result.answer
        # A ferramenta pedida na última rodada não roda (o resultado seria
        # descartado) e o pedido não fica pendurado no histórico.
        assert result.stats["tool_calls"] == MAX_TOOL_ITERATIONS
        pending = [m for m in result.conversation if getattr(m, "tool_calls", None)]
        assert len(pending) == MAX_TOOL_ITERATIONS

    def test_ferramenta_desconhecida_vira_mensagem_de_erro_nao_excecao(self, events):
        llm = FakeLLM([tool_round("ferramenta_inventada", {}), text_round("Desculpe.")])
        result = _run(llm, "x", events)
        assert "desconhecida" in result.trace[0]["result"].lower()
        assert result.answer == "Desculpe."

    def test_stream_vazio_devolve_mensagem_generica(self, events):
        result = _run(FakeLLM([[]]), "x", events)
        assert result.answer == GENERIC_MESSAGE

    def test_texto_em_blocos_ignora_raciocinio_interno(self, events):
        chunk = AIMessageChunk(
            content=[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Resposta."}]
        )
        result = _run(FakeLLM([[chunk]]), "x", events)
        assert result.answer == "Resposta."

    def test_erro_do_modelo_sobe_para_a_interface_tratar(self, events):
        with pytest.raises(RuntimeError):
            _run(FakeLLM([RuntimeError("429 RESOURCE_EXHAUSTED")]), "x", events)


class TestErros:
    @pytest.mark.parametrize(
        "exc,expected",
        [
            (AgentNotConfiguredError("sem chave"), NOT_CONFIGURED_MESSAGE),
            (RuntimeError("Error calling model (RESOURCE_EXHAUSTED): 429"), QUOTA_MESSAGE),
            (RuntimeError("You exceeded your current quota"), QUOTA_MESSAGE),
            (RuntimeError("API key not valid. PERMISSION_DENIED"), AUTH_MESSAGE),
            (TimeoutError("Read timed out"), NETWORK_MESSAGE),
            (ValueError("qualquer outra coisa"), GENERIC_MESSAGE),
        ],
    )
    def test_friendly_error(self, exc, expected):
        assert friendly_error(exc) == expected

    def test_sem_chave_levanta_erro_de_configuracao(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setattr(agent, "_llm_with_tools", None)
        with pytest.raises(AgentNotConfiguredError):
            agent._get_llm()


class TestAuxiliares:
    def test_route_devolve_so_as_ferramentas_da_primeira_decisao(self):
        decision = AIMessage(
            content="",
            tool_calls=[{"name": "predict_churn", "args": {}, "id": "1", "type": "tool_call"}],
        )
        llm = FakeLLM([], invoke_result=decision)
        with patch("src.agent._get_llm", return_value=llm):
            assert route("qual o risco?") == ["predict_churn"]
        assert len(llm.calls) == 1

    def test_chat_devolve_resposta_e_historico(self):
        with patch("src.agent._get_llm", return_value=FakeLLM([text_round("Oi!")])):
            answer, history = chat("olá")
        assert answer == "Oi!"
        assert history[-1].content == "Oi!"

    def test_cap_history_corta_na_fronteira_de_um_turno(self):
        conversation = []
        for i in range(10):
            conversation += [
                HumanMessage(content=f"pergunta {i}"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "t", "args": {}, "id": f"{i}", "type": "tool_call"}],
                ),
                ToolMessage(content="r", tool_call_id=f"{i}"),
                AIMessage(content=f"resposta {i}"),
            ]
        capped = cap_history(conversation, max_turns=3)
        assert sum(isinstance(m, HumanMessage) for m in capped) == 3
        assert capped[0].content == "pergunta 7"

    def test_cap_history_nao_mexe_quando_cabe(self):
        conversation = [HumanMessage(content="oi"), AIMessage(content="olá")]
        assert cap_history(conversation, max_turns=8) == conversation

    def test_perguntas_de_demonstracao(self):
        assert set(DEMO_PROMPTS) == {"prever", "fatores", "acoes"}
        for item in DEMO_PROMPTS.values():
            assert item["titulo"] and item["resumo"] and item["prompt"]

    def test_record_demo_grava_json_com_data_e_modelo(self, tmp_path):
        from src.record_demo import record

        def fake_runner(prompt):
            return TurnResult(
                answer=f"resposta para: {prompt[:10]}",
                conversation=[],
                trace=[
                    {
                        "tool": "search_churn_knowledge",
                        "args": {"query": "x"},
                        "result": "r",
                        "sources": ["a.md"],
                        "ms": 5,
                    }
                ],
                stats={"model_calls": 2},
            )

        path = tmp_path / "demo.json"
        payload = record(path=path, runner=fake_runner)
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved == payload
        assert saved["modelo"] and saved["gravado_em"]
        assert set(saved["respostas"]) == set(DEMO_PROMPTS)
        assert saved["respostas"]["fatores"]["ferramentas"][0]["sources"] == ["a.md"]

    def test_record_demo_salva_a_cada_resposta_e_retoma_de_onde_parou(self, tmp_path):
        """Com a cota acabando na 2ª pergunta, a 1ª (já paga) fica salva; a
        próxima execução grava só as que faltam, sem gastar cota de novo."""
        from src.record_demo import record

        path = tmp_path / "demo.json"
        calls = []

        def runner(prompt, fail_after=None):
            calls.append(prompt)
            if fail_after is not None and len(calls) > fail_after:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return TurnResult(answer="ok", conversation=[], stats={"model_calls": 1})

        with pytest.raises(RuntimeError):
            record(path=path, runner=lambda p: runner(p, fail_after=1))
        first = json.loads(path.read_text(encoding="utf-8"))
        assert list(first["respostas"]) == ["prever"]

        calls.clear()
        payload = record(path=path, runner=runner)
        assert len(calls) == len(DEMO_PROMPTS) - 1
        assert set(payload["respostas"]) == set(DEMO_PROMPTS)

        calls.clear()
        record(path=path, runner=runner)
        assert calls == []
        record(path=path, runner=runner, redo=True)
        assert len(calls) == len(DEMO_PROMPTS)

    def test_record_demo_nao_mistura_respostas_de_outro_modelo(self, tmp_path):
        from src.record_demo import record

        path = tmp_path / "demo.json"
        old = {"modelo": "modelo-antigo", "respostas": {"prever": {"resposta": "velha"}}}
        path.write_text(json.dumps(old), encoding="utf-8")
        calls = []

        def runner(prompt):
            calls.append(prompt)
            return TurnResult(answer="nova", conversation=[], stats={})

        payload = record(path=path, runner=runner)
        assert len(calls) == len(DEMO_PROMPTS)
        assert payload["respostas"]["prever"]["resposta"] == "nova"
