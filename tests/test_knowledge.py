"""A base de conhecimento do assistente acompanha o modelo em produção."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import Embeddings

import src.tools as tools
from src.build_knowledge_base import CHUNK_SIZE, build_index, load_chunks
from src.knowledge import GUIDE_DOC_PATH, MODEL_DOC_PATH, render_model_doc, render_reading_guide

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def artifacts():
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads((MODELS_DIR / "evaluation.json").read_text(encoding="utf-8"))
    return metrics, evaluation


def test_documento_do_modelo_esta_sincronizado_com_o_treino(artifacts):
    """Se falhar: rode `python -m src.train` (ele reescreve o documento) e
    depois `python -m src.build_knowledge_base`."""
    assert MODEL_DOC_PATH.read_text(encoding="utf-8") == render_model_doc(*artifacts)


def test_guia_de_leitura_esta_sincronizado_com_o_treino(artifacts):
    assert GUIDE_DOC_PATH.read_text(encoding="utf-8") == render_reading_guide(*artifacts)


def test_documento_cita_o_modelo_e_o_corte_em_producao(artifacts):
    metrics, evaluation = artifacts
    doc = render_model_doc(metrics, evaluation)
    assert f"{round(evaluation['optimal_threshold'] * 100)}%" in doc
    assert "regressão logística" in doc or "Random Forest" in doc or "Gradient Boosting" in doc


def test_trechos_levam_documento_e_secao_no_texto():
    chunks = load_chunks()
    assert len(chunks) >= 20
    for chunk in chunks:
        assert chunk.page_content.startswith("[")
        assert Path(chunk.metadata["source"]).exists()
        header, _, body = chunk.page_content.partition("]\n")
        assert body and len(body) <= CHUNK_SIZE


class FakeEmbeddings(Embeddings):
    """Embeddings determinísticos, sem rede nem cota do Gemini."""

    def __init__(self, *args, fail: bool = False, **kwargs):
        self.fail = fail

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class TestBuildIndex:
    def test_cria_a_base_e_nao_deixa_pastas_temporarias(self, tmp_path):
        target = str(tmp_path / "chroma_db")
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", FakeEmbeddings):
            n = build_index(target, api_key="fake")

        assert n == len(load_chunks())
        assert os.path.isfile(os.path.join(target, "chroma.sqlite3"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["chroma_db"]

    def test_falha_nos_embeddings_preserva_a_base_antiga(self, tmp_path):
        """O motivo da pasta temporária: com a cota do Gemini estourada no
        meio da reindexação, o assistente continua com a base que já tinha."""
        target = tmp_path / "chroma_db"
        target.mkdir()
        (target / "chroma.sqlite3").write_text("base antiga")

        def failing(*args, **kwargs):
            return FakeEmbeddings(fail=True)

        with (
            patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", failing),
            pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"),
        ):
            build_index(str(target), api_key="fake")

        assert (target / "chroma.sqlite3").read_text() == "base antiga"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["chroma_db"]

    def test_reindexar_substitui_sem_duplicar_trechos(self, tmp_path):
        from langchain_chroma import Chroma

        target = str(tmp_path / "chroma_db")
        with patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", FakeEmbeddings):
            build_index(target, api_key="fake")
            n = build_index(target, api_key="fake")

        store = Chroma(persist_directory=target, embedding_function=FakeEmbeddings())
        assert len(store.get()["ids"]) == n

    def test_sem_chave_avisa_em_vez_de_chamar_a_api(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            build_index(str(tmp_path / "chroma_db"))


class TestEnsureIndex:
    """A base vetorial não é versionada: na nuvem ela nasce na primeira busca."""

    @pytest.fixture
    def index_file(self, tmp_path, monkeypatch):
        path = tmp_path / "chroma_db" / "chroma.sqlite3"
        monkeypatch.setattr(tools, "CHROMA_INDEX_FILE", str(path))
        return path

    def test_base_pronta_nao_reconstroi(self, index_file):
        index_file.parent.mkdir()
        index_file.write_text("x")
        with patch("src.build_knowledge_base.build_index") as build:
            assert tools._ensure_index() is True
        build.assert_not_called()

    def test_sem_chave_nao_tenta_construir(self, index_file, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with patch("src.build_knowledge_base.build_index") as build:
            assert tools._ensure_index() is False
        build.assert_not_called()

    def test_com_chave_constroi_uma_vez(self, index_file, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")

        def fake_build():
            index_file.parent.mkdir(exist_ok=True)
            index_file.write_text("x")
            return 1

        with patch("src.build_knowledge_base.build_index", side_effect=fake_build) as build:
            assert tools._ensure_index() is True
            assert tools._ensure_index() is True
        build.assert_called_once()

    def test_falha_ao_construir_vira_aviso_para_o_modelo(self, index_file, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake")
        with patch("src.build_knowledge_base.build_index", side_effect=RuntimeError("cota")):
            message = tools.search_churn_knowledge.invoke(
                {
                    "name": "search_churn_knowledge",
                    "args": {"query": "churn"},
                    "id": "1",
                    "type": "tool_call",
                }
            )
        assert "não está disponível" in message.content
        assert message.artifact == {"sources": []}
