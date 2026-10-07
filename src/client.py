"""Cliente de previsão: API publicada, com plano B local.

A API roda no plano gratuito do Render, que hiberna o serviço depois de
15 min sem tráfego — o primeiro acesso seguinte leva cerca de 1 min. Para
quem abre o link não ficar olhando um spinner, este cliente:

1. tenta a API com timeout curto;
2. se ela não responder, calcula a MESMA previsão localmente — mesmo
   artefato de modelo, mesmo código de encoding e decisão
   (`src/predictor.py`) — e dispara, em segundo plano, uma chamada que
   acorda a API;
3. quando a API volta, as próximas previsões passam de novo por ela.

Toda resposta diz de onde veio (`source` = "api" ou "local"), e a interface
mostra isso ao visitante. Erros de validação (422) NUNCA caem no plano B:
um cliente inválido é inválido nos dois caminhos.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import requests
from pydantic import ValidationError

from src.schema import MAX_BATCH_SIZE, CustomerData

STATE_UNKNOWN = "verificando"
STATE_ONLINE = "online"
STATE_WAKING = "acordando"
STATE_OFFLINE = "offline"
STATE_LOCAL_ONLY = "local"


class InvalidCustomerError(ValueError):
    """Cliente recusado pela validação (a API responderia 422)."""

    def __init__(self, detail):
        super().__init__(str(detail))
        self.detail = detail


class ExplanationUnavailableError(RuntimeError):
    """Não há como explicar esta previsão agora (API sem /explain e modelo
    de árvore sem SHAP instalado localmente)."""


@dataclass(frozen=True)
class Answer:
    data: object
    source: str  # "api" ou "local"
    ms: float


def _validate(records: Sequence[Mapping]) -> list[dict]:
    """Mesmas regras da API (src/schema.py), com o erro no formato do 422."""
    validated = []
    for i, record in enumerate(records):
        try:
            validated.append(CustomerData(**record).model_dump())
        except ValidationError as exc:
            detail = exc.errors(include_url=False, include_context=False, include_input=False)
            for error in detail:
                error["loc"] = ["customers", i, *error.get("loc", ())]
            raise InvalidCustomerError(detail) from exc
    return validated


class ChurnClient:
    def __init__(
        self,
        api_url: str | None,
        timeout: float = 10.0,
        wake_timeout: float = 100.0,
        retry_after: float = 60.0,
    ):
        self.api_url = (api_url or "").strip().rstrip("/") or None
        self.timeout = timeout
        self.wake_timeout = wake_timeout
        self.retry_after = retry_after
        self._lock = threading.Lock()
        self._state = STATE_UNKNOWN if self.api_url else STATE_LOCAL_ONLY
        self._waking = False
        self._down_since = 0.0
        self._checked_at = 0.0
        self._latency_ms: float | None = None
        self._health: dict | None = None
        self._local = None

    # ------------------------------------------------------------------
    # Estado da API
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        with self._lock:
            return STATE_WAKING if self._waking else self._state

    def status(self) -> dict:
        with self._lock:
            return {
                "state": STATE_WAKING if self._waking else self._state,
                "latency_ms": self._latency_ms,
                "health": self._health,
                "api_url": self.api_url,
            }

    def _mark_up(self, latency_ms: float | None = None, health: dict | None = None) -> None:
        with self._lock:
            self._state = STATE_ONLINE
            self._checked_at = time.monotonic()
            if latency_ms is not None:
                self._latency_ms = latency_ms
            if health is not None:
                self._health = health

    def _mark_down(self) -> None:
        """A API não respondeu: segue no plano B e tenta acordá-la."""
        with self._lock:
            self._checked_at = time.monotonic()
            if self._waking:
                return
            self._waking = True
        threading.Thread(target=self._wake, name="acorda-api", daemon=True).start()

    def _wake(self) -> None:
        started = time.perf_counter()
        try:
            response = requests.get(f"{self.api_url}/health", timeout=self.wake_timeout)
            response.raise_for_status()
            health = response.json()
        except (requests.RequestException, ValueError):
            with self._lock:
                self._state = STATE_OFFLINE
                self._down_since = time.monotonic()
                self._waking = False
            return
        with self._lock:
            self._waking = False
        self._mark_up((time.perf_counter() - started) * 1000, health)

    def _remote_available(self) -> bool:
        if not self.api_url:
            return False
        with self._lock:
            if self._waking:
                return False
            if self._state == STATE_OFFLINE:
                return time.monotonic() - self._down_since >= self.retry_after
            return True

    def check(self, timeout: float = 1.5, max_age: float = 30.0) -> str:
        """Checagem rápida do /health, no máximo a cada `max_age` segundos.

        Com a API hibernando, o Render segura a conexão enquanto o serviço
        sobe; o timeout curto evita travar a página — a API é acordada em
        segundo plano e a interface segue com o plano B.
        """
        if not self.api_url:
            return STATE_LOCAL_ONLY
        with self._lock:
            fresh = time.monotonic() - self._checked_at < max_age
            if self._waking or (fresh and self._state != STATE_UNKNOWN):
                return STATE_WAKING if self._waking else self._state
        if not self._remote_available():
            return self.state
        started = time.perf_counter()
        try:
            response = requests.get(f"{self.api_url}/health", timeout=timeout)
            response.raise_for_status()
            health = response.json()
        except (requests.RequestException, ValueError):
            self._mark_down()
            return self.state
        self._mark_up((time.perf_counter() - started) * 1000, health)
        return STATE_ONLINE

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    def local(self):
        """Preditor local (carregado uma vez, sob demanda)."""
        with self._lock:
            if self._local is None:
                from src.predictor import ChurnPredictor

                self._local = ChurnPredictor()
            return self._local

    def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        started = time.perf_counter()
        response = requests.post(f"{self.api_url}{path}", json=payload, timeout=self.timeout)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code == 422:
            raise InvalidCustomerError(response.json().get("detail"))
        response.raise_for_status()
        return response.json(), elapsed

    def predict(self, records: Sequence[Mapping]) -> Answer:
        """Previsão de um ou mais clientes (lista de dicts crus)."""
        records = [dict(r) for r in records]
        if self._remote_available():
            try:
                predictions: list[dict] = []
                total_ms = 0.0
                for start in range(0, len(records), MAX_BATCH_SIZE):
                    chunk = records[start : start + MAX_BATCH_SIZE]
                    body, ms = self._post("/predict/batch", {"customers": chunk})
                    predictions.extend(body["predictions"])
                    total_ms += ms
                self._mark_up(total_ms if len(records) <= MAX_BATCH_SIZE else None)
                return Answer(predictions, "api", total_ms)
            except InvalidCustomerError:
                raise
            except (requests.RequestException, KeyError, ValueError):
                self._mark_down()

        started = time.perf_counter()
        predictor = self.local()
        data = predictor.predict(_validate(records))
        return Answer(data, "local", (time.perf_counter() - started) * 1000)

    def explain(self, record: Mapping) -> Answer:
        """Contribuição de cada variável para a previsão deste cliente."""
        record = dict(record)
        if self._remote_available():
            try:
                body, ms = self._post("/explain", record)
                self._mark_up(ms)
                return Answer(body, "api", ms)
            except InvalidCustomerError:
                raise
            except requests.HTTPError as exc:
                # 503 = explicabilidade desligada na API: o cálculo local
                # resolve se o modelo for linear (não precisa de SHAP).
                if exc.response is None or exc.response.status_code != 503:
                    self._mark_down()
            except (requests.RequestException, ValueError):
                self._mark_down()

        started = time.perf_counter()
        predictor = self.local()
        (validated,) = _validate([record])
        try:
            data = predictor.explain(validated)
        except ImportError as exc:
            raise ExplanationUnavailableError("SHAP não está instalado neste ambiente") from exc
        return Answer(data, "local", (time.perf_counter() - started) * 1000)