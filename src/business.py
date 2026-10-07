"""Regras de negócio puras (sem Streamlit, sem FastAPI) — testáveis isoladamente.

Tudo aqui é usado em mais de um lugar: `train.py` escolhe o threshold com as
mesmas funções que o Simulador de ROI usa para recalcular ao vivo, e a API e
a interface classificam o risco com a mesma regra. Uma regra, um lugar.

Modelo de valor usado no projeto
--------------------------------
Contatar um cliente custa `offer_cost` (a oferta de retenção). Um cliente que
ia cancelar e foi contatado aceita a oferta com probabilidade `success_rate`
e, nesse caso, a empresa preserva o `ltv` dele. Comparado a não fazer nada:

    valor líquido = TP × success_rate × LTV  −  (TP + FP) × offer_cost

(TP = contatados que iam cancelar; FP = contatados que não iam.) Os falsos
negativos não entram porque "não fazer nada" também os perde.

Com probabilidades calibradas, contatar vale a pena quando
p × success_rate × LTV > offer_cost, ou seja, p > offer_cost / (success_rate × LTV):
é o threshold teórico, que o threshold escolhido nos dados deve aproximar.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Hipóteses de referência — NÃO são medidas de uma empresa real. O Simulador
# de ROI deixa ajustar as três ao vivo.
DEFAULT_LTV = 1000.0
DEFAULT_OFFER_COST = 100.0
DEFAULT_SUCCESS_RATE = 0.30

# A partir de 50%, o cliente tem mais chance de cancelar do que de ficar.
HIGH_RISK_PROBABILITY = 0.5

RISK_LEVELS = ("Baixo", "Médio", "Alto")


# ---------------------------------------------------------------------------
# Valor, threshold e faixas de risco
# ---------------------------------------------------------------------------
def net_value(point: Mapping, ltv: float, offer_cost: float, success_rate: float) -> float:
    """Valor líquido de operar no ponto (tp/fp) de uma curva de threshold."""
    return point["tp"] * success_rate * ltv - (point["tp"] + point["fp"]) * offer_cost


def value_curve(
    curve: Sequence[Mapping], ltv: float, offer_cost: float, success_rate: float
) -> list[dict]:
    """Cada ponto da curva com o valor líquido e a taxa de contato calculados."""
    out = []
    for p in curve:
        n = p["tp"] + p["fp"] + p["fn"] + p["tn"]
        contacted = p["tp"] + p["fp"]
        out.append(
            {
                **p,
                "valor": net_value(p, ltv, offer_cost, success_rate),
                "contatados": contacted,
                "taxa_contato": contacted / n if n else 0.0,
                "recall": p["tp"] / (p["tp"] + p["fn"]) if (p["tp"] + p["fn"]) else 0.0,
                "precision": p["tp"] / contacted if contacted else 0.0,
            }
        )
    return out


def optimal_threshold(
    curve: Sequence[Mapping], ltv: float, offer_cost: float, success_rate: float
) -> float:
    """Threshold do ponto de maior valor líquido.

    Em empate, fica com o threshold MAIOR — mesmo valor com menos clientes
    contatados é a escolha mais conservadora (menos ofertas, menos atrito).
    """
    best = max(
        curve,
        key=lambda p: (net_value(p, ltv, offer_cost, success_rate), p["threshold"]),
    )
    return float(best["threshold"])


def theoretical_threshold(ltv: float, offer_cost: float, success_rate: float) -> float:
    """p mínima a partir da qual contatar tem valor esperado positivo."""
    if ltv <= 0 or success_rate <= 0:
        return 1.0
    return float(min(1.0, max(0.0, offer_cost / (success_rate * ltv))))


def risk_cuts(threshold: float) -> dict[str, float]:
    """Faixas de risco derivadas da decisão de negócio.

    - Baixo: abaixo do threshold — contatar custaria mais do que o esperado
      de retorno.
    - Médio: a partir do threshold — contatar compensa.
    - Alto: a partir de 50% — o cliente tem mais chance de cancelar do que
      de ficar (só faz sentido porque as probabilidades são calibradas).
    """
    return {
        "baixo_max": float(threshold),
        "medio_max": float(max(threshold, HIGH_RISK_PROBABILITY)),
    }


def risk_level(probability: float, cuts: Mapping[str, float]) -> str:
    if probability >= cuts["medio_max"]:
        return "Alto"
    if probability >= cuts["baixo_max"]:
        return "Médio"
    return "Baixo"


def expected_contact_value(
    probability: float, ltv: float, offer_cost: float, success_rate: float
) -> float:
    """Valor esperado de contatar ESTE cliente, comparado a não contatar."""
    return probability * success_rate * ltv - offer_cost


# ---------------------------------------------------------------------------
# Cenários "e se"
# ---------------------------------------------------------------------------
def whatif_scenarios(payload: Mapping) -> list[tuple[str, dict]]:
    """Mudanças de perfil que a empresa poderia oferecer a este cliente.

    Só entram cenários que mudam algo de verdade e que mantêm o cliente
    consistente (suporte técnico só para quem tem internet, etc.). O valor
    mostrado é a mudança na PREVISÃO do modelo — associação aprendida dos
    dados, não efeito causal garantido (ver aviso na interface).
    """
    scenarios: list[tuple[str, dict]] = []
    has_internet = payload.get("InternetService") != "No"

    if payload.get("Contract") != "One year":
        scenarios.append(("Migrar para contrato anual", {**payload, "Contract": "One year"}))
    if payload.get("Contract") != "Two year":
        scenarios.append(("Migrar para contrato bienal", {**payload, "Contract": "Two year"}))
    if has_internet and payload.get("TechSupport") == "No":
        scenarios.append(("Incluir suporte técnico", {**payload, "TechSupport": "Yes"}))
    if has_internet and payload.get("OnlineSecurity") == "No":
        scenarios.append(("Incluir segurança online", {**payload, "OnlineSecurity": "Yes"}))
    if payload.get("PaymentMethod") not in ("Bank transfer (automatic)", "Credit card (automatic)"):
        scenarios.append(
            (
                "Migrar para pagamento automático no cartão",
                {**payload, "PaymentMethod": "Credit card (automatic)"},
            )
        )
    return scenarios


# ---------------------------------------------------------------------------
# Drift — Population Stability Index
# ---------------------------------------------------------------------------
PSI_STABLE = 0.10
PSI_SHIFT = 0.25
PSI_MIN_ROWS = 100
_PSI_EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _PSI_EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _PSI_EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "Estável"
    if value < PSI_SHIFT:
        return "Atenção"
    return "Mudança forte"


def build_reference_profile(
    df: pd.DataFrame, categorical: Iterable[str], numeric: Iterable[str], bins: int = 10
) -> dict:
    """Perfil do conjunto de TREINO usado como referência de drift.

    Categóricas: proporção de cada categoria. Numéricas: bordas de decis (os
    bins têm ~10% dos clientes de treino cada) e a proporção em cada bin.
    """
    profile: dict = {"n": int(len(df)), "categorical": {}, "numeric": {}}
    for col in categorical:
        shares = df[col].astype(str).value_counts(normalize=True)
        profile["categorical"][col] = {str(k): float(v) for k, v in shares.items()}
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        profile["numeric"][col] = {
            "edges": [float(x) for x in inner],
            "shares": [float(x) for x in counts / counts.sum()],
        }
    return profile


def drift_report(batch: pd.DataFrame, profile: Mapping) -> list[dict]:
    """PSI de cada variável do lote contra o perfil de referência."""
    rows = []
    for col, ref in profile.get("categorical", {}).items():
        if col not in batch.columns:
            continue
        categories = list(ref)
        actual_shares = batch[col].astype(str).value_counts(normalize=True)
        expected = np.array([ref[c] for c in categories])
        actual = np.array([actual_shares.get(c, 0.0) for c in categories])
        value = _psi(expected, actual)
        rows.append({"variavel": col, "psi": value, "status": psi_status(value)})
    for col, ref in profile.get("numeric", {}).items():
        if col not in batch.columns:
            continue
        values = pd.to_numeric(batch[col], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            continue
        inner = np.asarray(ref["edges"])
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        value = _psi(np.asarray(ref["shares"]), counts / counts.sum())
        rows.append({"variavel": col, "psi": value, "status": psi_status(value)})
    return sorted(rows, key=lambda r: r["psi"], reverse=True)


# ---------------------------------------------------------------------------
# Formatação em português do Brasil
# ---------------------------------------------------------------------------
def _br(text: str) -> str:
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_int(value: float) -> str:
    return _br(f"{round(value):,}")


def fmt_num(value: float, decimals: int = 2) -> str:
    return _br(f"{value:,.{decimals}f}")


def fmt_pct(value: float, decimals: int = 1) -> str:
    return _br(f"{value * 100:,.{decimals}f}") + "%"


def fmt_pp(value: float, decimals: int = 1) -> str:
    """Diferença de probabilidade em pontos percentuais, com sinal."""
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    return f"{sign}{_br(f'{abs(value) * 100:,.{decimals}f}')} pp"


def fmt_brl(value: float, cents: bool = False) -> str:
    decimals = 2 if cents else 0
    text = _br(f"{abs(value):,.{decimals}f}")
    return f"{'−' if value < 0 else ''}R$ {text}"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
