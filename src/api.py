"""API REST de previsão de churn (FastAPI).

Sobe com: `uvicorn src.api:app --reload` a partir da raiz do projeto.

Camada HTTP fina: validação em `src/schema.py`, previsão e decisão em
`src/predictor.py`. Aqui ficam só rotas, limites e observabilidade.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.predictor import ChurnPredictor
from src.schema import (
    MAX_BATCH_SIZE,
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerData,
    ExplanationResponse,
    PredictionResponse,
)

__all__ = ["MAX_BATCH_SIZE", "CustomerData", "app"]

logger = logging.getLogger("churn_api")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

predictor = ChurnPredictor()
model = predictor.model
expected_columns: list[str] = predictor.columns
DECISION_THRESHOLD: float = predictor.threshold
RISK_LEVEL_CUTS: dict[str, float] = predictor.cuts

# Modelos lineares são explicados de forma exata, sem SHAP. Para modelos de
# árvore o SHAP é importado sob demanda (~100 MB de RAM) — ENABLE_SHAP_EXPLAIN
# desliga /explain sem redeploy se a memória do free tier apertar.
ENABLE_SHAP_EXPLAIN = os.getenv("ENABLE_SHAP_EXPLAIN", "true").lower() == "true"

# Limite simples por IP (janela deslizante de 60 s), sem dependência externa.
# 0 desliga. Atrás de proxy (Render), o IP real vem de X-Forwarded-For.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
_RATE_LIMITED_PATHS = {"/predict", "/predict/batch", "/explain"}
_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(
    title="Churn Radar API",
    description=(
        "Prevê a probabilidade calibrada de um cliente de telecom cancelar, com "
        "threshold de decisão e faixas de risco derivados de uma política de "
        "retenção (valor esperado de contatar vs. não contatar)."
    ),
    version=__version__,
)

# CORS: a interface Streamlit chama a API pelo servidor, não pelo navegador,
# e não depende disto. Consumidores que rodam NO navegador (um front-end JS)
# precisam desses headers. A API é pública, só leitura e sem cookies.
CORS_ALLOW_ORIGINS = [
    origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "desconhecido"


@app.middleware("http")
async def rate_limit_and_log(request: Request, call_next):
    if RATE_LIMIT_PER_MINUTE > 0 and request.url.path in _RATE_LIMITED_PATHS:
        now = time.monotonic()
        hits = _hits[_client_ip(request)]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={"detail": "Muitas requisições. Tente de novo em um minuto."},
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        if len(_hits) > 10_000:  # evita crescer sem limite com IPs antigos
            for ip in [ip for ip, q in _hits.items() if not q]:
                del _hits[ip]

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    if request.url.path != "/health":
        logger.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed_ms, 1),
                }
            )
        )
    return response


@app.get("/")
def root() -> dict:
    return {"status": "online", "message": "Churn Radar API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    """Usado pelo healthcheck do Render e pela interface para saber se o
    serviço já saiu da hibernação do free tier — e qual modelo está no ar."""
    return {
        "status": "healthy",
        "api_version": __version__,
        "model_selected": predictor.model_selected,
        "trained_at": predictor.trained_at,
        "decision_threshold": DECISION_THRESHOLD,
        "risk_level_cuts": RISK_LEVEL_CUTS,
        "model_features": len(expected_columns),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData) -> PredictionResponse:
    try:
        result = predictor.predict([data.model_dump()])[0]
    except Exception:
        # O payload já passou pelo Pydantic: uma falha aqui é do servidor
        # (modelo corrompido, versão incompatível), não do cliente.
        logger.exception("Falha ao gerar previsão")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar a previsão.") from None
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Mesma lógica de /predict, vetorizada: um único predict_proba para o
    lote inteiro. Com categorias fixas no encoding, o resultado de cada
    cliente é o mesmo que ele teria sozinho — não depende dos vizinhos."""
    try:
        results = predictor.predict([c.model_dump() for c in request.customers])
    except Exception:
        logger.exception("Falha ao gerar previsão em lote")
        raise HTTPException(
            status_code=500, detail="Erro interno ao gerar as previsões em lote."
        ) from None
    return BatchPredictResponse(predictions=[PredictionResponse(**r) for r in results])


@app.post("/explain", response_model=ExplanationResponse)
def explain(data: CustomerData) -> ExplanationResponse:
    """Quanto cada variável contribuiu para a previsão deste cliente,
    ordenado por magnitude. Exato para a regressão logística; SHAP para
    modelos de árvore."""
    if not predictor.is_linear and not ENABLE_SHAP_EXPLAIN:
        raise HTTPException(
            status_code=503,
            detail="Explicabilidade desativada nesta instância (ENABLE_SHAP_EXPLAIN=false).",
        )
    try:
        result = predictor.explain(data.model_dump())
    except ImportError:
        # Modelo de árvore sem o SHAP instalado na imagem (ver
        # requirements-api.txt): indisponível, não erro interno.
        raise HTTPException(
            status_code=503,
            detail="Explicabilidade indisponível nesta instância (SHAP não instalado).",
        ) from None
    except Exception:
        logger.exception("Falha ao gerar explicação")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar a explicação.") from None
    return ExplanationResponse(**result)