import os
import requests
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

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
        response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Não foi possível conectar à API de previsão. Verifique se ela está rodando. Detalhe: {str(e)}"}


persist_directory = os.path.join(
    os.path.dirname(__file__), "..", "chroma_db"
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBEDDING_MODEL"),
    google_api_key=api_key
)

vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def search_churn_knowledge(query: str) -> str:
    """
    Busca informações na base de conhecimento sobre churn: o que é churn,
    fatores de risco, recomendações de retenção e detalhes sobre o modelo.
    Use esta ferramenta para perguntas abertas/conceituais sobre churn,
    NÃO para prever um cliente específico.
    """
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)