"""Ferramentas do agente: previsão (function calling) e busca no RAG."""

from __future__ import annotations

import os
from contextlib import suppress

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.client import ChurnClient, ExplanationUnavailableError, InvalidCustomerError
from src.labels import field_contributions

load_dotenv()

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("CHURN_API_TIMEOUT", "10"))
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

_client: ChurnClient | None = None
_retriever = None


def get_client() -> ChurnClient:
    """Cliente compartilhado (API com plano B local — ver src/client.py).

    A interface injeta o dela com set_client(), para o agente e as telas
    enxergarem o mesmo estado da API.
    """
    global _client
    if _client is None:
        _client = ChurnClient(API_URL, timeout=REQUEST_TIMEOUT)
    return _client


def set_client(client: ChurnClient) -> None:
    global _client
    _client = client


def _top_factors(explanation: dict, record: dict, limit: int = 5) -> list[dict]:
    """As características que mais pesaram, em português e com a direção."""
    rows = field_contributions(explanation.get("contributions", []), record)
    return [
        {
            "fator": f"{row['label']}: {row['value']}",
            "efeito": "aumenta o risco" if row["contribution"] > 0 else "reduz o risco",
        }
        for row in rows[:limit]
        if abs(row["contribution"]) > 1e-9
    ]


@tool(response_format="content_and_artifact")
def predict_churn(
    gender: str,
    SeniorCitizen: int,
    Partner: str,
    Dependents: str,
    tenure: int,
    PhoneService: str,
    MultipleLines: str,
    InternetService: str,
    OnlineSecurity: str,
    OnlineBackup: str,
    DeviceProtection: str,
    TechSupport: str,
    StreamingTV: str,
    StreamingMovies: str,
    Contract: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: float,
) -> tuple[dict, dict]:
    """
    Prevê a probabilidade de um cliente específico cancelar o serviço (churn)
    e devolve os fatores que mais pesaram na previsão.

    Use quando o usuário descrever um cliente específico e perguntar sobre o
    risco dele. Os valores precisam estar em inglês, exatamente como no
    dataset (ex.: "Fiber optic", "Month-to-month", "Electronic check").
    Regras de consistência: sem internet (InternetService="No") exige
    "No internet service" nos 6 serviços de internet; sem telefone exige
    MultipleLines="No phone service"; TotalCharges deve ficar perto de
    tenure × MonthlyCharges. Se faltar alguma informação essencial, pergunte
    ao usuário em vez de inventar.
    """
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    client = get_client()
    try:
        answer = client.predict([payload])
    except InvalidCustomerError as exc:
        # O detalhe da validação volta para o modelo: é com ele que o
        # agente corrige o argumento errado ou pergunta ao usuário.
        return {
            "error": "Valores inválidos ou inconsistentes para o modelo.",
            "detail": exc.detail,
        }, {"fonte": None}
    except Exception as exc:
        return {"error": f"Não foi possível gerar a previsão agora: {exc}"}, {"fonte": None}

    result = dict(answer.data[0])
    # A explicação é um "extra": se falhar, a previsão continua valendo.
    with suppress(ExplanationUnavailableError, InvalidCustomerError, RuntimeError):
        result["principais_fatores"] = _top_factors(client.explain(payload).data, payload)
    return result, {"fonte": answer.source, "ms": round(answer.ms)}


def _get_retriever():
    """Inicializa o retriever sob demanda — importar este módulo não exige a
    chave do Gemini nem uma base vetorial pronta."""
    global _retriever
    if _retriever is None:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return _retriever


@tool(response_format="content_and_artifact")
def search_churn_knowledge(query: str) -> tuple[str, dict]:
    """
    Busca na base de conhecimento do projeto: o que é churn, fatores de risco,
    ações de retenção, como o modelo foi construído e como ler o risco.
    Use para perguntas abertas/conceituais — NÃO para prever um cliente.
    """
    # response_format="content_and_artifact": `content` é o que o modelo lê;
    # `artifact` viaja no ToolMessage sem entrar no contexto do LLM — é de
    # onde a interface tira os nomes dos documentos-fonte.
    if not os.path.isdir(CHROMA_DIR):
        return (
            "A base de conhecimento ainda não foi construída. "
            "Rode `python -m src.build_knowledge_base` para indexá-la.",
            {"sources": []},
        )

    docs = _get_retriever().invoke(query)
    if not docs:
        return "Nenhum documento relevante encontrado na base de conhecimento.", {"sources": []}

    # dict.fromkeys preserva a ordem de relevância removendo duplicatas.
    sources = list(
        dict.fromkeys(os.path.basename(doc.metadata.get("source", "desconhecido")) for doc in docs)
    )
    content = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return content, {"sources": sources}
