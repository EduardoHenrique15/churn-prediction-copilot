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

import gc
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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


class _Precomputed(Embeddings):
    """Entrega ao Chroma os vetores já calculados, sem nova chamada à API."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("só usado na indexação")


def build_index(target: str = CHROMA_DIR, api_key: str | None = None) -> int:
    """Indexa os .md em `target` e devolve o número de trechos.

    A base nova é montada numa pasta temporária e só substitui a antiga no
    final: se os embeddings falharem no meio (cota, rede), a base que já
    existia continua intacta. Indexar direto na pasta definitiva também não
    serve porque Chroma.from_documents ACRESCENTA à coleção existente — rodar
    de novo duplicaria os trechos.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY não configurada")
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"Nenhum documento .md encontrado em {KNOWLEDGE_BASE_DIR}")

    from chromadb.api.client import SharedSystemClient
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Os embeddings são calculados ANTES de criar qualquer arquivo: é a etapa
    # que falha na prática (cota, rede), e assim uma falha não deixa nada no
    # disco — no Windows, uma pasta com o Chroma aberto nem poderia ser apagada.
    embedder = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    texts = [chunk.page_content for chunk in chunks]
    vectors = dict(zip(texts, embedder.embed_documents(texts), strict=True))

    staging = f"{target}.novo"
    shutil.rmtree(staging, ignore_errors=True)
    store = Chroma.from_documents(
        documents=chunks, embedding=_Precomputed(vectors), persist_directory=staging
    )
    # Fecha os arquivos da pasta temporária antes de renomeá-la: no Windows,
    # renomear uma pasta com arquivos abertos falha.
    del store
    SharedSystemClient.clear_system_cache()
    gc.collect()
    # A base antiga sai do caminho com um rename, que é tudo ou nada: no
    # Windows, se a interface estiver com ela aberta, o PermissionError sobe
    # sem estragar nada (um rmtree direto poderia apagar só metade).
    retired = f"{target}.antiga"
    shutil.rmtree(retired, ignore_errors=True)
    if os.path.isdir(target):
        os.replace(target, retired)
    os.replace(staging, target)
    shutil.rmtree(retired, ignore_errors=True)
    return len(chunks)


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit(
            "GEMINI_API_KEY não configurada. Defina-a no arquivo .env "
            "antes de construir a base de conhecimento."
        )
    try:
        n_chunks = build_index()
    except PermissionError as exc:
        raise SystemExit(
            f"Não foi possível substituir a base antiga em {CHROMA_DIR} — ela está "
            "aberta por outro processo. Feche a interface (streamlit) e rode de novo."
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"{n_chunks} trechos indexados em: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
