"""Ferramentas do agente: previsão (function calling) e busca no RAG."""

from __future__ import annotations

import logging
import os
import threading
from contextlib import suppress

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.client import ChurnClient, ExplanationUnavailableError, InvalidCustomerError
from src.labels import field_contributions
from src.utils import (
    ContractType,
    Gender,
    InternetServiceType,
    PaymentMethodType,
    YesNo,
    YesNoInternet,
    YesNoPhone,
)

load_dotenv()

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("CHURN_API_TIMEOUT", "10"))
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
# O Chroma grava o índice neste arquivo. Testar o arquivo (e não só a pasta)
# evita tratar como pronta uma pasta vazia ou de uma indexação interrompida.
CHROMA_INDEX_FILE = os.path.join(CHROMA_DIR, "chroma.sqlite3")

logger = logging.getLogger(__name__)

_client: ChurnClient | None = None
_retriever = None
_index_lock = threading.Lock()


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
    # Os tipos Literal (os mesmos do schema da API, em src/utils.py) viram
    # listas de valores permitidos no schema que o Gemini recebe: o modelo
    # escolhe entre "Fiber optic" e "DSL" em vez de adivinhar a grafia — e
    # erra menos, sem gastar uma rodada (e cota) para corrigir o argumento.
    gender: Gender,
    SeniorCitizen: int,
    Partner: YesNo,
    Dependents: YesNo,
    tenure: int,
    PhoneService: YesNo,
    MultipleLines: YesNoPhone,
    InternetService: InternetServiceType,
    OnlineSecurity: YesNoInternet,
    OnlineBackup: YesNoInternet,
    DeviceProtection: YesNoInternet,
    TechSupport: YesNoInternet,
    StreamingTV: YesNoInternet,
    StreamingMovies: YesNoInternet,
    Contract: ContractType,
    PaperlessBilling: YesNo,
    PaymentMethod: PaymentMethodType,
    MonthlyCharges: float,
    TotalCharges: float,
) -> tuple[dict, dict]:
    """
    Prevê a probabilidade de um cliente específico cancelar o serviço (churn)
    e devolve os fatores que mais pesaram na previsão.

    Use quando o usuário descrever um cliente específico e perguntar sobre o
    risco dele. Os valores categóricos estão em inglês, como no dataset, e
    só aceitam as opções listadas. SeniorCitizen é 1 (idoso) ou 0.
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


def _ensure_index() -> bool:
    """Garante a base vetorial, construindo-a na primeira busca se faltar.

    A pasta chroma_db/ não vai para o git: no deploy (Streamlit Cloud,
    Docker) ela é montada aqui, uma vez, a partir dos .md versionados em
    data/knowledge_base. Custa uma requisição de embeddings — e só acontece
    quando o assistente já está em uso, ou seja, com a chave configurada.
    O lock impede duas sessões de indexarem ao mesmo tempo.
    """
    if os.path.isfile(CHROMA_INDEX_FILE):
        return True
    if not os.getenv("GEMINI_API_KEY"):
        return False
    with _index_lock:
        if not os.path.isfile(CHROMA_INDEX_FILE):
            from src.build_knowledge_base import build_index

            build_index()
    return os.path.isfile(CHROMA_INDEX_FILE)


def _invalid_arguments(error) -> str:
    """Valor fora das opções (o schema já as lista, mas o modelo pode errar):
    volta para ele como texto curto, para corrigir na próxima rodada."""
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
        for e in getattr(error, "errors", lambda: [])()
    )
    return f"Valores inválidos para o modelo — {problems or error}"


predict_churn.handle_validation_error = _invalid_arguments


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
    try:
        ready = _ensure_index()
    except Exception:
        logger.exception("Falha ao construir a base de conhecimento")
        ready = False
    if not ready:
        return (
            "A base de conhecimento não está disponível agora. Responda com o que você "
            "sabe e avise que não consultou os documentos do projeto. (Para indexá-la: "
            "`python -m src.build_knowledge_base`.)",
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
