"""A base de conhecimento do assistente acompanha o modelo em produção."""

import json
from pathlib import Path

import pytest

from src.build_knowledge_base import CHUNK_SIZE, load_chunks
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
