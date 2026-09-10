import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from tools import predict_churn, search_churn_knowledge

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
    temperature=0
)

tools = [predict_churn, search_churn_knowledge]
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {t.name: t for t in tools}

SYSTEM_PROMPT = """Você é um assistente especializado em análise de churn de clientes
de telecomunicações. Você tem duas ferramentas disponíveis:

1. predict_churn: use quando o usuário fornecer dados completos de um cliente
   específico e quiser saber a probabilidade de cancelamento dele.
2. search_churn_knowledge: use para perguntas conceituais sobre churn, fatores
   de risco, recomendações de retenção ou informações sobre o modelo.

Responda sempre em português, de forma clara e direta, explicando os resultados
em linguagem natural (não apenas retornando números crus)."""


def chat(user_message: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]

    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    while ai_response.tool_calls:
        for tool_call in ai_response.tool_calls:
            selected_tool = tools_by_name[tool_call["name"]]
            tool_result = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

    content = ai_response.content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content