"""Dados da interface: artefatos do treino, exemplos e o cliente de previsão.

Tudo aqui é cacheado: os arquivos só mudam quando o modelo é retreinado (e
o app, republicado).
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.client import ChurnClient
from src.ui.settings import API_TIMEOUT, API_URL

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
DEMO_PATH = DATA_DIR / "demo" / "agent_demo.json"

EXAMPLE_FILES = {
    "tipica": ("exemplo_lote.csv", "Base típica", "200 clientes reais do conjunto de teste"),
    "novos": (
        "exemplo_lote_novos.csv",
        "Clientes recém-chegados",
        "200 clientes com até 12 meses de casa — perfil bem diferente do treino",
    ),
}


@st.cache_resource(show_spinner=False)
def get_client() -> ChurnClient:
    """Um cliente para o app inteiro (e para o agente): o estado da API —
    online, acordando, fora do ar — é o mesmo para todas as sessões."""
    from src import tools

    client = ChurnClient(API_URL, timeout=API_TIMEOUT)
    tools.set_client(client)
    return client


@st.cache_data(show_spinner=False)
def _json(path: str) -> dict:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def metrics() -> dict:
    return _json(str(MODELS_DIR / "metrics.json"))


def evaluation() -> dict:
    return _json(str(MODELS_DIR / "evaluation.json"))


def reference_profile() -> dict:
    return _json(str(MODELS_DIR / "reference_profile.json"))


def demo_answers() -> dict:
    return _json(str(DEMO_PATH))


@st.cache_data(show_spinner=False)
def test_predictions() -> np.ndarray:
    """Probabilidades previstas para os clientes de TESTE (fora da amostra)."""
    path = MODELS_DIR / "test_predictions.csv"
    if not path.exists():
        return np.array([])
    return pd.read_csv(path)["y_proba"].to_numpy()


@st.cache_data(show_spinner=False)
def example_batch(key: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / EXAMPLE_FILES[key][0])


# ---------------------------------------------------------------------------
# Uso do assistente (compartilhado entre sessões: a cota do Gemini é uma só)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _agent_usage() -> dict:
    return {"lock": threading.Lock(), "date": None, "count": 0, "quota_date": None}


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def agent_questions_today() -> int:
    usage = _agent_usage()
    with usage["lock"]:
        if usage["date"] != _today():
            usage["date"], usage["count"] = _today(), 0
        return usage["count"]


def register_agent_question() -> None:
    usage = _agent_usage()
    with usage["lock"]:
        if usage["date"] != _today():
            usage["date"], usage["count"] = _today(), 0
        usage["count"] += 1


def mark_quota_exhausted() -> None:
    usage = _agent_usage()
    with usage["lock"]:
        usage["quota_date"] = _today()


def quota_exhausted() -> bool:
    return _agent_usage()["quota_date"] == _today()
