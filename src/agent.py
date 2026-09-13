"""Agente conversacional de churn (LangChain + Gemini).

Combina duas capacidades: function calling para prever um cliente específico
via API e RAG para perguntas conceituais sobre churn.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools import predict_churn, search_churn_knowledge

load_dotenv()

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

MAX_TOOL_ITERATIONS = 6

SYSTEM_PROMPT = """Você é um assistente especializado em análise de churn de clientes
de telecomunicações. Você tem duas ferramentas disponíveis:

1. predict_churn: use quando o usuário fornecer dados completos de um cliente
   específico e quiser saber a probabilidade de cancelamento dele.
2. search_churn_knowledge: use para perguntas conceituais sobre churn, fatores
   de risco, recomendações de retenção ou informações sobre o modelo.

Responda sempre em português, de forma clara e direta, explicando os resultados
em linguagem natural (não apenas retornando números crus)."""

TOOLS = [predict_churn, search_churn_knowledge]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_llm_with_tools = None


def _get_llm():
    """Instancia o modelo sob demanda, com erro legível se faltar a chave."""
    global _llm_with_tools
    if _llm_with_tools is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY não configurada. Defina-a no arquivo .env "
                "(ou nos secrets do Streamlit) para usar o agente."
            )
        llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL, google_api_key=api_key, temperature=0
        )
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


def _extract_text(content) -> str:
    """O Gemini pode devolver o conteúdo como string ou como lista de blocos."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content or ""


def chat(
    user_message: str, history: list[BaseMessage] | None = None
) -> tuple[str, list[BaseMessage]]:
    """Processa uma mensagem e devolve (resposta, histórico atualizado).

    O histórico é devolvido para que o chamador (a interface ou o terminal) o
    guarde e reenvie na próxima chamada. Sem isso cada pergunta começaria do
    zero e o agente não entenderia perguntas de acompanhamento como
    "e o que eu faço com esse cliente?".
    """
    llm = _get_llm()

    conversation: list[BaseMessage] = list(history or [])
    conversation.append(HumanMessage(content=user_message))

    ai_response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *conversation])
    conversation.append(ai_response)

    iterations = 0
    while isinstance(ai_response, AIMessage) and ai_response.tool_calls:
        if iterations >= MAX_TOOL_ITERATIONS:
            fallback = (
                "Não consegui concluir a análise: o número máximo de consultas "
                "às ferramentas foi atingido. Tente reformular a pergunta."
            )
            conversation.append(AIMessage(content=fallback))
            return fallback, conversation

        for tool_call in ai_response.tool_calls:
            selected_tool = TOOLS_BY_NAME.get(tool_call["name"])
            if selected_tool is None:
                result = f"Ferramenta desconhecida: {tool_call['name']}"
            else:
                try:
                    result = selected_tool.invoke(tool_call["args"])
                except Exception as exc:
                    result = f"A ferramenta falhou: {exc}"
            conversation.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

        ai_response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *conversation])
        conversation.append(ai_response)
        iterations += 1

    return _extract_text(ai_response.content), conversation