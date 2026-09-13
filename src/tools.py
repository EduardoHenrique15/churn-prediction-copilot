"""Ferramentas do agente: previsão via API (function calling) e busca no RAG."""

import os

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")

REQUEST_TIMEOUT = int(os.getenv("CHURN_API_TIMEOUT", "60"))

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db"
)

_retriever = None


@tool
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
) -> dict:
    """
    Prevê a probabilidade de um cliente específico cancelar o serviço (churn),
    com base em seus dados de perfil, serviços contratados e conta.
    Use esta ferramenta quando o usuário fornecer dados de um cliente específico
    e perguntar sobre o risco de cancelamento dele.
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

    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException as exc:
        return {
            "error": (
                f"Não foi possível conectar à API de previsão em {API_URL}. "
                f"Verifique se ela está rodando. Detalhe: {exc}"
            )
        }

    if response.status_code == 422:
        return {
            "error": "Valores inválidos para o modelo.",
            "detail": response.json().get("detail"),
        }

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        return {"error": f"A API retornou erro: {exc}"}

    return response.json()


def _get_retriever():
    """Inicializa o retriever sob demanda.

    Feito de forma preguiçosa para que importar este módulo não exija a chave
    do Gemini nem uma base vetorial pronta — importante porque a interface
    importa as ferramentas antes de saber se o usuário vai usar o chat.
    """
    global _retriever
    if _retriever is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, embedding_function=embeddings
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever


@tool
def search_churn_knowledge(query: str) -> str:
    """
    Busca informações na base de conhecimento sobre churn: o que é churn,
    fatores de risco, recomendações de retenção e detalhes sobre o modelo.
    Use esta ferramenta para perguntas abertas/conceituais sobre churn,
    NÃO para prever um cliente específico.
    """
    if not os.path.isdir(CHROMA_DIR):
        return (
            "A base de conhecimento ainda não foi construída. "
            "Rode `python -m src.build_knowledge_base` para indexá-la."
        )

    docs = _get_retriever().invoke(query)
    if not docs:
        return "Nenhum documento relevante encontrado na base de conhecimento."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)