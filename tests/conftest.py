"""Fixtures compartilhadas pelos testes.

Nenhum teste daqui chama a rede: a API é testada em processo (TestClient),
o agente usa um modelo de linguagem falso e o cliente de previsão roda no
modo local. Os testes que usam o Gemini de verdade estão marcados como
`integration` e ficam fora do `pytest` padrão (ver pyproject.toml).
"""

from __future__ import annotations

import pytest

from src.schema import EXAMPLE_CUSTOMER

NO_INTERNET_SERVICES = {
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
}


def make_customer(**overrides) -> dict:
    """Cliente válido (o exemplo do schema) com alterações pontuais."""
    return {**EXAMPLE_CUSTOMER, **overrides}


def loyal_customer() -> dict:
    """Perfil de baixíssimo risco: 70 meses, contrato bienal, sem internet."""
    return make_customer(
        **NO_INTERNET_SERVICES,
        tenure=70,
        Contract="Two year",
        PaymentMethod="Bank transfer (automatic)",
        PaperlessBilling="No",
        MonthlyCharges=20.0,
        TotalCharges=1400.0,
    )


@pytest.fixture
def customer() -> dict:
    return make_customer()


@pytest.fixture
def local_client():
    """Cliente de previsão sem API: calcula tudo localmente."""
    from src.client import ChurnClient

    return ChurnClient(api_url=None)