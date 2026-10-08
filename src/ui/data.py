"""Dados da interface: artefatos do treino, exemplos e o cliente de previsão.

Tudo aqui é cacheado: os arquivos só mudam quando o modelo é retreinado (e
o app, republicado).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import streamlit as st

from src.business import (
    DEFAULT_LTV,
    DEFAULT_OFFER_COST,
    DEFAULT_SUCCESS_RATE,
    optimal_threshold,
    risk_cuts,
)
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
# Hipóteses da campanha (compartilhadas entre Estratégia e Carteira)
# ---------------------------------------------------------------------------
# Chaves dos controles da página Estratégia. Com persist_state="session", o
# valor sobrevive à troca de página — é assim que a Carteira enxerga o que a
# pessoa ajustou lá. A taxa de sucesso fica em % (o controle é inteiro).
COST_KEYS = {"ltv": "s_ltv", "offer_cost": "s_offer", "success_pct": "s_success"}


def reference_costs() -> dict[str, float]:
    """Hipóteses de referência do treino (models/evaluation.json)."""
    costs = evaluation().get("cost_assumptions", {})
    return {
        "ltv": float(costs.get("ltv", DEFAULT_LTV)),
        "offer_cost": float(costs.get("offer_cost", DEFAULT_OFFER_COST)),
        "success_rate": float(costs.get("success_rate", DEFAULT_SUCCESS_RATE)),
    }


def campaign_policy() -> dict:
    """Hipóteses em uso nesta sessão e a decisão que sai delas.

    Sem ajuste na Estratégia, valem as de referência — e o corte e as faixas
    são exatamente os da API (o treino escolhe o corte com a mesma função,
    sobre as mesmas previsões fora da amostra). Com ajuste, o corte é
    recalculado aqui, do mesmo jeito que a Estratégia mostra.
    """
    ref = reference_costs()
    state = st.session_state
    costs = {
        "ltv": float(state.get(COST_KEYS["ltv"], ref["ltv"])),
        "offer_cost": float(state.get(COST_KEYS["offer_cost"], ref["offer_cost"])),
        "success_rate": float(
            state.get(COST_KEYS["success_pct"], round(ref["success_rate"] * 100)) / 100
        ),
    }
    custom = any(abs(costs[k] - ref[k]) > 1e-9 for k in ref)
    ev = evaluation()
    if custom and ev.get("curves"):
        threshold = optimal_threshold(
            ev["curves"]["oof"], costs["ltv"], costs["offer_cost"], costs["success_rate"]
        )
        cuts = risk_cuts(threshold)
    else:
        threshold = float(ev.get("optimal_threshold", 0.5))
        cuts = ev.get("risk_level_cuts") or risk_cuts(threshold)
    return {"costs": costs, "threshold": threshold, "cuts": cuts, "custom": custom}


# ---------------------------------------------------------------------------
# Uso do assistente (compartilhado entre sessões: a cota do Gemini é uma só)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _agent_usage() -> dict:
    return {"lock": threading.Lock(), "date": None, "count": 0, "quota_date": None}


def _quota_timezone() -> tzinfo:
    """A cota gratuita do Gemini zera à meia-noite do Pacífico, não à do UTC.

    Contar o dia em UTC liberava perguntas ~8 h antes de a cota voltar: a
    primeira falhava e a página só então marcava a cota como esgotada. Sem a
    base de fusos (Windows sem o pacote tzdata), usa UTC−8 fixo — erra por
    no máximo 1 h no horário de verão, em vez de 8 h.
    """
    try:
        return ZoneInfo("America/Los_Angeles")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=-8))


def _today() -> str:
    return datetime.now(_quota_timezone()).date().isoformat()


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
