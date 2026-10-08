# API de previsão (FastAPI) — publicada no Render (plano gratuito, 512 MB).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Lista enxuta: sem Streamlit, LangChain nem SHAP (ver requirements-api.txt).
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Só os módulos da raiz de src/ — a interface (src/ui/) não entra na imagem.
COPY src/*.py ./src/
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '8000')}/health\")"

# `exec` troca o shell pelo uvicorn: ele passa a ser o PID 1 e recebe o
# SIGTERM do Render direto, encerrando de forma limpa. O shell só existe
# para expandir ${PORT}, que o Render define em tempo de execução.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
