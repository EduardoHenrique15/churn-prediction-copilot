"""Indexa a base de conhecimento no ChromaDB para o RAG do agente.

Rode a partir da raiz do projeto: `python -m src.build_knowledge_base`

Usa a API de embeddings do Gemini (algumas requisições, uma por lote de
trechos). Rode de novo sempre que um .md de data/knowledge_base mudar —
inclusive depois de `python -m src.train`, que reescreve os documentos 04 e
05 com os números do modelo novo.

Os documentos são cortados por seção (títulos "#" e "##") e cada trecho
leva o nome do documento e da seção no começo do texto: o trecho recuperado
chega ao modelo com o contexto de onde veio.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.tools import CHROMA_DIR, EMBEDDING_MODEL

load_dotenv()

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"

# Uma seção inteira cabe num trecho na maioria dos casos (as seções têm até
# ~700 caracteres); as mais longas são quebradas com sobreposição.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def load_chunks(directory: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    """Lê os .md e devolve os trechos prontos para indexar."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "documento"), ("##", "secao")], strip_headers=True
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for section in size_splitter.split_documents(sections):
            title = section.metadata.get("documento", path.stem)
            part = section.metadata.get("secao")
            context = f"{title} — {part}" if part else title
            chunks.append(
                Document(
                    page_content=f"[{context}]\n{section.page_content}",
                    metadata={"source": str(path), "secao": part or ""},
                )
            )
    return chunks


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY não configurada. Defina-a no arquivo .env "
            "antes de construir a base de conhecimento."
        )

    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"Nenhum documento .md encontrado em {KNOWLEDGE_BASE_DIR}")
    n_docs = len({c.metadata["source"] for c in chunks})
    print(f"{n_docs} documentos, {len(chunks)} trechos")

    # Reindexação limpa: Chroma.from_documents ACRESCENTA à coleção que já
    # existe no diretório. Sem apagar a base antiga, rodar este script de novo
    # duplicaria os trechos (e manteria versões velhas dos documentos).
    if os.path.isdir(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError as exc:
            raise SystemExit(
                f"Não foi possível apagar a base antiga em {CHROMA_DIR} — ela está "
                "aberta por outro processo. Feche a interface (streamlit) e rode de novo."
            ) from exc
        print("Base vetorial anterior removida (reindexação limpa)")

    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    Chroma.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key),
        persist_directory=CHROMA_DIR,
    )
    print(f"Base vetorial criada em: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
