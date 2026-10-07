"""Configurações da interface: secrets do Streamlit, variáveis de ambiente
ou o padrão, nessa ordem."""

from __future__ import annotations

import os


def setting(name: str, default: str = "") -> str:
    """Lê uma configuração sem quebrar quando não existe secrets.toml.

    `st.secrets` levanta exceção se o arquivo não existir — o caso normal na
    máquina de quem desenvolve.
    """
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(setting(name, str(default)))
    except ValueError:
        return default


def _export(*names: str) -> None:
    """Copia secrets para o ambiente: o agente (src/agent.py) lê variáveis de
    ambiente, e no Streamlit Community Cloud as chaves ficam nos secrets."""
    for name in names:
        value = setting(name)
        if value and not os.getenv(name):
            os.environ[name] = value


_export("GEMINI_API_KEY", "GEMINI_CHAT_MODEL", "GEMINI_EMBEDDING_MODEL")

API_URL = setting("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT = float(setting("CHURN_API_TIMEOUT", "10") or 10)

AUTHOR_NAME = setting("AUTHOR_NAME") or "Eduardo Henrique"
GITHUB_URL = (
    setting("GITHUB_URL") or "https://github.com/EduardoHenrique15/churn-prediction-copilot"
)
LINKEDIN_URL = setting("LINKEDIN_URL") or "https://www.linkedin.com/in/eduardo-henrique15/"

# Limites do assistente na demonstração pública: a cota gratuita do Gemini
# é de ~20 requisições por dia, e cada pergunta usa 1 ou 2.
AGENT_MAX_QUESTIONS_PER_SESSION = _int("AGENT_MAX_QUESTIONS_PER_SESSION", 3)
AGENT_MAX_QUESTIONS_PER_DAY = _int("AGENT_MAX_QUESTIONS_PER_DAY", 6)
