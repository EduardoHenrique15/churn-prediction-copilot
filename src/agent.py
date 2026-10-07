"""Agente conversacional de churn (LangChain + Gemini).

Combina duas capacidades: function calling para prever um cliente específico
(via API) e RAG para perguntas conceituais sobre churn e sobre o projeto.

Cada rodada é UMA chamada ao modelo, em streaming: o texto chega token a
token e, se o modelo pedir uma ferramenta, as partes da chamada são
agregadas, a ferramenta roda e a próxima rodada começa. Não existe chamada
extra só para "transmitir" a resposta — importante com a cota gratuita do
Gemini (~20 requisições/dia).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import message_chunk_to_message

from src.tools import predict_churn, search_churn_knowledge

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

# Rodadas de ferramenta por pergunta. Cada rodada é uma requisição ao Gemini;
# na prática uma pergunta usa 1 (sem ferramenta) ou 2 (com ferramenta).
MAX_TOOL_ITERATIONS = 3

# Turnos (pergunta + tudo até a resposta) mantidos no histórico enviado ao
# modelo — sem teto, cada pergunta reenviaria a conversa inteira.
MAX_HISTORY_TURNS = 8

SYSTEM_PROMPT = """Você é o assistente de retenção do Churn Radar, um projeto que prevê o
cancelamento de clientes de telecomunicações. Responda sempre em português do Brasil,
de forma clara, direta e sem jargão desnecessário.

Ferramentas:
1. predict_churn — use quando o usuário descrever um cliente específico e quiser saber o
   risco. Converta a descrição para os valores em inglês do dataset. Se faltar informação
   essencial, pergunte antes de chamar. Nunca invente números.
2. search_churn_knowledge — use para perguntas sobre churn, fatores de risco, ações de
   retenção, como o modelo funciona ou como ler o risco.

Ao explicar uma previsão: diga a probabilidade, a faixa de risco, se vale a pena contatar
(churn_prediction = true significa que contatar tem valor esperado positivo) e os fatores
que mais pesaram. Lembre que o modelo mede associações nos dados, não causas.
Se a pergunta não tiver relação com churn, retenção de clientes ou com este projeto,
diga educadamente que esse não é o seu escopo."""

TOOLS = [predict_churn, search_churn_knowledge]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Perguntas de exemplo mostradas na interface (e gravadas pelo
# src/record_demo.py). A de previsão traz os 19 campos para o agente não
# precisar pedir dados que faltam.
DEMO_PROMPTS: dict[str, dict[str, str]] = {
    "prever": {
        "titulo": "Prever um cliente",
        "resumo": "Cliente de 3 meses, contrato mensal, fibra óptica, sem suporte técnico, "
        "R$ 95/mês — qual o risco de cancelar?",
        "prompt": "Cliente mulher, não idosa, sem cônjuge, sem dependentes, 3 meses de casa, "
        "com telefone, sem múltiplas linhas, internet fibra óptica, sem segurança online, "
        "sem backup online, sem proteção de aparelho, sem suporte técnico, com streaming de "
        "TV e de filmes, contrato mensal, fatura digital, pagamento por cheque eletrônico, "
        "mensalidade de 95 reais, total gasto de 285 reais. Qual o risco de cancelamento?",
    },
    "fatores": {
        "titulo": "Entender o churn",
        "resumo": "Quais são os principais fatores de risco de churn em telecom?",
        "prompt": "Quais são os principais fatores de risco de churn em telecom?",
    },
    "acoes": {
        "titulo": "Pedir recomendações",
        "resumo": "Que ações de retenção funcionam melhor para clientes de contrato mensal?",
        "prompt": "Que ações de retenção funcionam melhor para clientes de contrato mensal?",
    },
}


class AgentNotConfiguredError(RuntimeError):
    """A chave do Gemini não está configurada nesta instância."""


QUOTA_MESSAGE = (
    "O assistente atingiu o limite de uso da cota gratuita da API do Gemini. "
    "Tente novamente mais tarde — enquanto isso, as respostas de demonstração "
    "continuam disponíveis."
)
NOT_CONFIGURED_MESSAGE = (
    "O assistente não está configurado nesta instância (falta a chave da API do Gemini)."
)
AUTH_MESSAGE = "A chave da API do Gemini foi recusada. Verifique a configuração do assistente."
NETWORK_MESSAGE = "Não consegui falar com o serviço de IA agora. Tente de novo em instantes."
GENERIC_MESSAGE = "Não consegui responder agora. Tente de novo em instantes."


def is_quota_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("resource_exhausted", "429", "quota", "ratelimit", "rate limit"))


def friendly_error(exc: BaseException) -> str:
    """Mensagem para o visitante — o erro técnico vai só para o log."""
    if isinstance(exc, AgentNotConfiguredError):
        return NOT_CONFIGURED_MESSAGE
    if is_quota_error(exc):
        return QUOTA_MESSAGE
    text = f"{type(exc).__name__} {exc}".lower()
    if any(
        k in text for k in ("api key", "api_key", "permission_denied", "unauthenticated", "401")
    ):
        return AUTH_MESSAGE
    if any(k in text for k in ("timeout", "timed out", "connection", "unavailable", "503")):
        return NETWORK_MESSAGE
    return GENERIC_MESSAGE


_llm_with_tools = None


def _get_llm():
    """Instancia o modelo sob demanda (importar o módulo não exige a chave)."""
    global _llm_with_tools
    if _llm_with_tools is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise AgentNotConfiguredError("GEMINI_API_KEY não configurada")
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key)
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


def _extract_text(content) -> str:
    """O Gemini pode devolver o conteúdo como string ou como lista de blocos."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    return content or ""


def _run_tool_call(tool_call: dict) -> tuple[ToolMessage, dict]:
    """Executa uma tool call e devolve (ToolMessage para o histórico, registro
    para a trilha de transparência mostrada na interface)."""
    selected_tool = TOOLS_BY_NAME.get(tool_call["name"])
    started = time.perf_counter()
    if selected_tool is None:
        message = ToolMessage(
            content=f"Ferramenta desconhecida: {tool_call['name']}",
            tool_call_id=tool_call["id"],
        )
    else:
        try:
            # Invocar com o tool_call COMPLETO devolve um ToolMessage pronto
            # (e, para o RAG, com `.artifact` trazendo as fontes).
            message = selected_tool.invoke(tool_call)
        except Exception as exc:
            message = ToolMessage(
                content=f"A ferramenta falhou: {exc}", tool_call_id=tool_call["id"]
            )

    artifact = getattr(message, "artifact", None)
    artifact = artifact if isinstance(artifact, dict) else {}
    record = {
        "tool": tool_call["name"],
        "args": tool_call["args"],
        "result": message.content,
        # RAG: documentos consultados. Previsão: se veio da API ou do plano B.
        "sources": artifact.get("sources"),
        "fonte": artifact.get("fonte"),
        "ms": round((time.perf_counter() - started) * 1000),
    }
    return message, record


@dataclass
class TurnResult:
    answer: str
    conversation: list[BaseMessage]
    trace: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def run_turn(
    user_message: str,
    history: list[BaseMessage] | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> TurnResult:
    """Responde a uma mensagem, uma chamada ao modelo por rodada, em streaming.

    `on_event` recebe, em ordem:
    - {"type": "text", "text": ...} a cada pedaço de texto;
    - {"type": "reset"} se a rodada terminou pedindo ferramentas (o texto
      parcial daquela rodada, se houve, não é a resposta final);
    - {"type": "tool", "record": {...}} depois de cada ferramenta executada.

    Devolve a resposta final, o histórico atualizado, a trilha de
    ferramentas e estatísticas (latência, chamadas ao modelo, tokens).
    """
    emit = on_event or (lambda _event: None)
    llm = _get_llm()
    conversation: list[BaseMessage] = list(history or [])
    conversation.append(HumanMessage(content=user_message))
    trace: list[dict] = []
    stats = {"model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
    started = time.perf_counter()

    for _ in range(MAX_TOOL_ITERATIONS + 1):
        aggregated = None
        for chunk in llm.stream([SystemMessage(content=SYSTEM_PROMPT), *conversation]):
            aggregated = chunk if aggregated is None else aggregated + chunk
            piece = _extract_text(chunk.content)
            if piece:
                emit({"type": "text", "text": piece})
        stats["model_calls"] += 1

        if aggregated is None:
            answer = GENERIC_MESSAGE
            conversation.append(AIMessage(content=answer))
            break

        usage = getattr(aggregated, "usage_metadata", None) or {}
        stats["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        stats["output_tokens"] += int(usage.get("output_tokens", 0) or 0)

        message = message_chunk_to_message(aggregated)
        conversation.append(message)
        if not getattr(message, "tool_calls", None):
            answer = _extract_text(message.content) or GENERIC_MESSAGE
            break

        emit({"type": "reset"})
        for tool_call in message.tool_calls:
            tool_message, record = _run_tool_call(tool_call)
            conversation.append(tool_message)
            trace.append(record)
            stats["tool_calls"] += 1
            emit({"type": "tool", "record": record})
    else:
        answer = (
            "Não consegui concluir a análise: o número máximo de consultas às "
            "ferramentas foi atingido. Tente reformular a pergunta."
        )
        conversation.append(AIMessage(content=answer))

    stats["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return TurnResult(answer=answer, conversation=conversation, trace=trace, stats=stats)


def route(user_message: str) -> list[str]:
    """Só a PRIMEIRA decisão do modelo: quais ferramentas ele chamaria.

    Uma requisição ao Gemini e nenhuma ferramenta executada. É o que o eval
    de roteamento (tests/test_agent_eval.py) mede — pela metade da cota que
    custaria rodar o turno inteiro.
    """
    message = _get_llm().invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)]
    )
    return [call["name"] for call in getattr(message, "tool_calls", None) or []]


def cap_history(
    conversation: list[BaseMessage], max_turns: int = MAX_HISTORY_TURNS
) -> list[BaseMessage]:
    """Mantém só os últimos `max_turns` turnos — sempre cortando no início de
    um turno (HumanMessage), nunca no meio: uma ToolMessage órfã (sem a
    AIMessage que a pediu) é rejeitada pela API do Gemini."""
    human_indices = [i for i, m in enumerate(conversation) if isinstance(m, HumanMessage)]
    if len(human_indices) <= max_turns:
        return conversation
    return conversation[human_indices[-max_turns] :]


def chat(
    user_message: str, history: list[BaseMessage] | None = None
) -> tuple[str, list[BaseMessage]]:
    """Versão sem interface (terminal): devolve (resposta, histórico)."""
    result = run_turn(user_message, history)
    return result.answer, cap_history(result.conversation)
