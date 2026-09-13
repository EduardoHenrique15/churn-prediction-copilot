"""Indexa a base de conhecimento no ChromaDB para o RAG do agente.

Rode a partir da raiz do projeto: `python -m src.build_knowledge_base`
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.tools import CHROMA_DIR, EMBEDDING_MODEL

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY não configurada. Defina-a no arquivo .env "
            "antes de construir a base de conhecimento."
        )

    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        raise SystemExit(f"Nenhum documento .md encontrado em {KNOWLEDGE_BASE_DIR}")
    print(f"{len(documents)} documentos carregados")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(documents)
    print(f"{len(chunks)} chunks gerados")

    Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=api_key
        ),
        persist_directory=CHROMA_DIR,
    )

    print(f"Banco vetorial criado em: {CHROMA_DIR}")
    print(f"{len(chunks)} chunks indexados com sucesso")


if __name__ == "__main__":
    main()