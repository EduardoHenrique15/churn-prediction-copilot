"""Churn Radar — interface Streamlit do projeto de previsão de churn.

Rode a partir da raiz do projeto: `streamlit run app.py`

Nota sobre HTML em Streamlit
----------------------------
`st.markdown(..., unsafe_allow_html=True)` passa a string por um parser de
Markdown ANTES de renderizar o HTML. Isso cria duas armadilhas clássicas que
fazem o markup aparecer como texto cru na tela:

1. qualquer linha com 4+ espaços de indentação vira bloco de código;
2. uma linha em branco no meio do HTML quebra o bloco em dois parágrafos.

Todo HTML dinâmico deste arquivo passa por `ui_html()`, que remove a
indentação de cada linha e junta tudo numa linha só — as duas armadilhas
deixam de existir por construção, não por disciplina de quem edita depois.

O CSS fica em `STYLES`, uma constante que NÃO é f-string: as chaves do CSS
conflitam com a interpolação de f-strings e essa é a outra origem comum do
mesmo bug.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
import streamlit as st

from src.utils import CATEGORY_OPTIONS

st.set_page_config(
    page_title="Churn Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def _setting(name: str, default: str) -> str:
    """Lê uma configuração dos secrets, do ambiente ou do default.

    `st.secrets` levanta exceção quando não existe `.streamlit/secrets.toml`,
    o que é o caso normal na máquina do desenvolvedor — por isso o acesso é
    protegido em vez de assumir que o arquivo existe.
    """
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

API_URL = _setting("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")

API_TIMEOUT = 60

RISK_COLORS = {"Alto": "#F87171", "Médio": "#FBBF24", "Baixo": "#34D399"}

_SIM_NAO = {"Yes": "Sim", "No": "Não"}
_SEM_INTERNET = {**_SIM_NAO, "No internet service": "Sem internet"}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "gender": {"Female": "Feminino", "Male": "Masculino"},
    "MultipleLines": {**_SIM_NAO, "No phone service": "Sem telefone"},
    "InternetService": {
        "DSL": "DSL",
        "Fiber optic": "Fibra óptica",
        "No": "Sem internet",
    },
    "Contract": {
        "Month-to-month": "Mensal",
        "One year": "Anual (1 ano)",
        "Two year": "Bienal (2 anos)",
    },
    "PaymentMethod": {
        "Electronic check": "Cheque eletrônico",
        "Mailed check": "Cheque por correio",
        "Bank transfer (automatic)": "Transferência bancária (automática)",
        "Credit card (automatic)": "Cartão de crédito (automático)",
    },
}
VALUE_LABELS.update(
    dict.fromkeys(
        ("Partner", "Dependents", "PhoneService", "PaperlessBilling"), _SIM_NAO
    )
)
VALUE_LABELS.update(
    dict.fromkeys(
        (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ),
        _SEM_INTERNET,
    )
)


def pt(field: str):
    """format_func do selectbox: exibe em português, devolve o valor original."""
    return lambda value: VALUE_LABELS.get(field, {}).get(value, value)


def ui_html(markup: str) -> None:
    """Renderiza HTML imune ao parser de Markdown do Streamlit.

    Ver a nota no topo do módulo: remove a indentação de cada linha e colapsa
    tudo numa linha só.
    """
    compact = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
--bg: #05070D;
--surface: #0B1120;
--surface-2: #0F172A;
--border: rgba(34, 211, 238, 0.14);
--cyan: #22D3EE;
--indigo: #818CF8;
--text: #E2E8F0;
--muted: #64748B;
--ok: #34D399;
--warn: #FBBF24;
--danger: #F87171;
}

.stApp { background: var(--bg); }

#MainMenu, footer, header { visibility: hidden; }

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* ---------- Hero ---------- */
.hero { position: relative; border: 1px solid var(--border); border-radius: 18px; padding: 30px 34px; margin-bottom: 18px; background: radial-gradient(120% 160% at 12% 0%, rgba(34,211,238,0.13) 0%, rgba(129,140,248,0.07) 38%, rgba(5,7,13,0) 70%), var(--surface); overflow: hidden; }
.hero::after { content: ''; position: absolute; inset: 0; background-image: linear-gradient(rgba(34,211,238,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.05) 1px, transparent 1px); background-size: 34px 34px; -webkit-mask-image: radial-gradient(70% 100% at 20% 0%, #000 0%, transparent 75%); mask-image: radial-gradient(70% 100% at 20% 0%, #000 0%, transparent 75%); pointer-events: none; }
.hero-badge { position: relative; z-index: 1; display: inline-flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--cyan); border: 1px solid var(--border); background: rgba(34,211,238,0.07); padding: 5px 12px; border-radius: 999px; margin-bottom: 16px; }
.hero h1 { position: relative; z-index: 1; margin: 0 0 10px 0; font-size: 44px; line-height: 1.06; font-weight: 700; letter-spacing: -0.028em; background: linear-gradient(96deg, #F8FAFC 12%, var(--cyan) 52%, var(--indigo) 92%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { position: relative; z-index: 1; margin: 0; color: #94A3B8; font-size: 15px; max-width: 62ch; line-height: 1.62; }

/* ---------- HUD ---------- */
.hud { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
.hud-item { flex: 1 1 170px; display: flex; align-items: center; gap: 10px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface); padding: 11px 15px; }
.hud-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); display: block; margin-bottom: 3px; }
.hud-value { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--text); }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-ok { background: var(--ok); box-shadow: 0 0 0 4px rgba(52,211,153,0.14); animation: pulse 2.2s ease-in-out infinite; }
.dot-off { background: var(--danger); box-shadow: 0 0 0 4px rgba(248,113,113,0.14); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ---------- Cards ---------- */
.card { border: 1px solid var(--border); border-radius: 16px; background: var(--surface); padding: 24px 26px; height: 100%; }
.card-title { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--cyan); margin: 0 0 4px 0; }
.card-sub { color: var(--muted); font-size: 12.5px; margin: 0 0 20px 0; }

/* ---------- Radar ---------- */
.radar-wrap { display: flex; justify-content: center; padding: 14px 0 4px 0; }
.sweep { transform-origin: 110px 110px; animation: sweep 3.6s linear infinite; }
@keyframes sweep { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.blip { animation: blip 3.6s ease-in-out infinite; }
@keyframes blip { 0%, 62% { opacity: 0; } 72% { opacity: 1; } 100% { opacity: 0; } }

/* ---------- Gauge ---------- */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.gauge-arc { transition: stroke-dashoffset 1.05s cubic-bezier(0.22, 1, 0.36, 1); }
.gauge-value { font-family: 'JetBrains Mono', monospace; font-size: 46px; font-weight: 700; letter-spacing: -0.03em; }
.gauge-caption { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }
.risk-chip { display: inline-flex; align-items: center; gap: 7px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 6px 15px; border-radius: 999px; margin-top: 10px; }

/* ---------- Drivers ---------- */
.driver { display: flex; align-items: flex-start; gap: 11px; padding: 11px 0; border-bottom: 1px solid rgba(148,163,184,0.09); }
.driver:last-child { border-bottom: none; }
.driver-icon { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; width: 20px; flex-shrink: 0; text-align: center; margin-top: 1px; }
.driver-text { font-size: 13.5px; color: #CBD5E1; line-height: 1.5; }
.driver-text b { color: #F1F5F9; font-weight: 600; }

/* ---------- Telemetria ---------- */
.metric-row { display: flex; justify-content: space-between; align-items: baseline; padding: 7px 0; }
.metric-name { font-size: 12px; color: var(--muted); }
.metric-val { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 600; color: var(--text); }
.bar-track { height: 5px; border-radius: 3px; background: rgba(148,163,184,0.12); overflow: hidden; margin-top: 5px; }
.bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--cyan), var(--indigo)); }
.feat-name { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #94A3B8; }
.feat-val { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--cyan); }

/* ---------- Empty state ---------- */
.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 26px 20px 34px 20px; }
.empty h4 { margin: 16px 0 6px 0; font-size: 15.5px; color: #CBD5E1; font-weight: 600; }
.empty p { margin: 0; font-size: 13px; color: var(--muted); max-width: 34ch; line-height: 1.6; }

/* ---------- Skeleton ---------- */
.sk { border-radius: 10px; background: linear-gradient(90deg, rgba(148,163,184,0.07) 25%, rgba(148,163,184,0.16) 37%, rgba(148,163,184,0.07) 63%); background-size: 400% 100%; animation: shimmer 1.35s ease-in-out infinite; }
@keyframes shimmer { 0% { background-position: 100% 50%; } 100% { background-position: 0 50%; } }
.sk-circle { width: 210px; height: 210px; border-radius: 50%; margin: 0 auto 22px auto; }
.sk-line { height: 12px; margin-bottom: 11px; }

/* ---------- Chat ---------- */
/* min-height em vez de height:100%: as colunas do Streamlit não esticam os
   filhos, então a altura percentual não equalizaria os três cards. */
.suggestion { border: 1px solid var(--border); border-radius: 11px; background: var(--surface-2); padding: 13px 16px; font-size: 13px; color: #CBD5E1; line-height: 1.5; min-height: 118px; }
.suggestion b { color: var(--cyan); font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.13em; text-transform: uppercase; display: block; margin-bottom: 6px; }

/* ---------- Formulário ---------- */
.form-section { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.17em; text-transform: uppercase; color: var(--cyan); margin: 20px 0 10px 0; padding-bottom: 7px; border-bottom: 1px solid var(--border); }

/* Os rótulos dos widgets são pintados pelo tema do Streamlit, não por este
   CSS. Se o .streamlit/config.toml não for carregado, o Streamlit assume tema
   claro e escreve os rótulos em cinza-escuro sobre o nosso fundo escuro —
   ilegível. Fixar a cor aqui garante contraste mesmo nesse cenário. */
/* O seletor é por atributo, sem prefixo de tag: stWidgetLabel é um <label>,
   e o texto fica dois níveis abaixo, dentro de um <p>. */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * { color: #CBD5E1 !important; }
[data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 500 !important; }

[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #94A3B8 !important; }

/* O fundo dos selects/sliders vem do tema do Streamlit, não daqui: no 1.63
   esses elementos não expõem data-testid nem data-baseweb, só uma classe
   st-emotion-cache-* gerada por build e instável entre versões. Por isso o
   .streamlit/config.toml é obrigatório — perseguir essas classes seria um
   hack que quebraria no próximo update. As regras acima garantem que, mesmo
   sem o config, o texto continue legível. */
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #64748B !important; }

/* ---------- Controles do Streamlit ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 42px; background: transparent; color: var(--muted); font-size: 13.5px; font-weight: 500; padding: 0 18px; }
.stTabs [aria-selected="true"] { color: var(--cyan); border-bottom: 2px solid var(--cyan); }
/* O botão de envio do formulário tem test-id próprio: sem ele na lista, o
   botão principal da tela fica com o estilo padrão do Streamlit. */
.stButton > button, div[data-testid="stFormSubmitButton"] > button { width: 100%; border-radius: 10px; border: 1px solid rgba(34,211,238,0.35); background: linear-gradient(96deg, rgba(34,211,238,0.15), rgba(129,140,248,0.15)); color: var(--cyan); font-weight: 600; font-size: 13.5px; letter-spacing: 0.03em; padding: 11px 0; transition: transform 0.14s ease, box-shadow 0.18s ease, border-color 0.18s ease; }
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { border-color: var(--cyan); box-shadow: 0 0 22px rgba(34,211,238,0.22); transform: translateY(-1px); color: #ECFEFF; }
.stButton > button:active, div[data-testid="stFormSubmitButton"] > button:active { transform: translateY(0); }
div[data-testid="stForm"] { border: 1px solid var(--border); border-radius: 16px; background: var(--surface); padding: 20px 22px; }
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
div[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--border); border-radius: 13px; }
div[data-testid="stChatInput"] { border: 1px solid var(--border); border-radius: 12px; background: var(--surface-2); }
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color: #CBD5E1 !important; }
div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] h4,
div[data-testid="stChatMessage"] strong, div[data-testid="stChatMessage"] b { color: #F1F5F9 !important; }
div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3 { font-size: 17px !important; margin-top: 18px !important; }
div[data-testid="stChatMessage"] code { color: var(--cyan) !important; background: rgba(34,211,238,0.09) !important; }
div[data-testid="stChatMessage"] a { color: var(--cyan) !important; }

/* ---------- Responsivo ---------- */
@media (max-width: 860px) {
.hero { padding: 24px 20px; }
.hero h1 { font-size: 31px; }
.hero p { font-size: 13.5px; }
.card { padding: 19px 17px; }
.block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Dados e comunicação com a API
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Telemetria do último treino, regravada por `python -m src.train`."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data(ttl=20, show_spinner=False)
def check_api() -> tuple[bool, float | None, int | None]:
    """Estado da API. Cacheado por 20s para não bater no serviço a cada rerun."""
    try:
        started = time.perf_counter()
        response = requests.get(f"{API_URL}/health", timeout=8)
        latency = (time.perf_counter() - started) * 1000
        if response.ok:
            return True, latency, response.json().get("model_features")
    except requests.exceptions.RequestException:
        pass
    return False, None, None


def call_predict(payload: dict) -> tuple[dict | None, str | None]:
    """Chama /predict e devolve (resultado, erro)."""
    try:
        response = requests.post(
            f"{API_URL}/predict", json=payload, timeout=API_TIMEOUT
        )
    except requests.exceptions.Timeout:
        return None, (
            "A API não respondeu a tempo. No free tier do Render o primeiro "
            "acesso acorda o serviço e pode levar até um minuto — tente de novo."
        )
    except requests.exceptions.RequestException:
        return None, f"Não foi possível conectar à API em {API_URL}."

    if response.status_code == 422:
        return None, "A API rejeitou os dados enviados (validação do schema)."
    if not response.ok:
        return None, f"A API retornou o status {response.status_code}."
    return response.json(), None


def risk_drivers(data: dict) -> list[tuple[str, str]]:
    """Fatores de risco e de proteção presentes no perfil informado.

    Regras derivadas das features de maior importância do modelo (tenure,
    contrato, tipo de internet, forma de pagamento e serviços de proteção).
    É leitura de perfil, não explicabilidade por previsão — atribuição real
    por caso entra quando o SHAP for adicionado.
    """
    drivers: list[tuple[str, str]] = []

    if data["tenure"] < 12:
        drivers.append(("risk", f"Cliente novo: <b>{data['tenure']} meses</b> de casa — a faixa de maior churn."))
    elif data["tenure"] >= 48:
        drivers.append(("safe", f"Cliente maduro: <b>{data['tenure']} meses</b> de relacionamento."))

    if data["Contract"] == "Month-to-month":
        drivers.append(("risk", "Contrato <b>mensal</b>, sem fidelidade — pode sair a qualquer momento."))
    else:
        drivers.append(("safe", f"Contrato de <b>{'1 ano' if data['Contract'] == 'One year' else '2 anos'}</b> reduz muito o risco."))

    if data["InternetService"] == "Fiber optic":
        drivers.append(("risk", "Internet <b>fibra óptica</b>: maior ticket e maior churn no dataset."))

    if data["PaymentMethod"] == "Electronic check":
        drivers.append(("risk", "Pagamento por <b>cheque eletrônico</b>, associado à maior taxa de cancelamento."))

    sem_protecao = [
        nome
        for campo, nome in (
            ("OnlineSecurity", "segurança online"),
            ("TechSupport", "suporte técnico"),
            ("DeviceProtection", "proteção de aparelho"),
        )
        if data[campo] == "No"
    ]
    if sem_protecao:
        drivers.append(("risk", f"Sem <b>{'</b>, <b>'.join(sem_protecao)}</b> — serviços que aumentam a retenção."))

    if data["MonthlyCharges"] >= 80:
        drivers.append(("risk", f"Mensalidade alta: <b>R$ {data['MonthlyCharges']:.2f}</b>."))

    return drivers


# --------------------------------------------------------------------------
# Componentes visuais
# --------------------------------------------------------------------------
def render_radar(active: bool = False) -> None:
    color = "#22D3EE" if active else "#334155"
    ui_html(
        f"""
        <div class="radar-wrap">
        <svg width="220" height="220" viewBox="0 0 220 220" role="img" aria-label="Radar de varredura">
        <defs>
        <linearGradient id="sweepGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="{color}" stop-opacity="0.38"/>
        <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </linearGradient>
        </defs>
        <circle cx="110" cy="110" r="98" fill="none" stroke="{color}" stroke-opacity="0.18" stroke-width="1"/>
        <circle cx="110" cy="110" r="70" fill="none" stroke="{color}" stroke-opacity="0.15" stroke-width="1"/>
        <circle cx="110" cy="110" r="42" fill="none" stroke="{color}" stroke-opacity="0.13" stroke-width="1"/>
        <circle cx="110" cy="110" r="14" fill="none" stroke="{color}" stroke-opacity="0.22" stroke-width="1"/>
        <line x1="110" y1="12" x2="110" y2="208" stroke="{color}" stroke-opacity="0.1" stroke-width="1"/>
        <line x1="12" y1="110" x2="208" y2="110" stroke="{color}" stroke-opacity="0.1" stroke-width="1"/>
        <g class="sweep"><path d="M110 110 L110 12 A98 98 0 0 1 179 41 Z" fill="url(#sweepGrad)"/></g>
        <circle class="blip" cx="152" cy="74" r="3.5" fill="{color}"/>
        <circle class="blip" cx="78" cy="141" r="2.5" fill="{color}" style="animation-delay:1.2s"/>
        <circle class="blip" cx="137" cy="156" r="2.5" fill="{color}" style="animation-delay:2.1s"/>
        <circle cx="110" cy="110" r="3" fill="{color}"/>
        </svg>
        </div>
        """
    )


def render_gauge(probability: float, risk: str) -> None:
    """Gauge circular. O arco usa stroke-dasharray sobre a circunferência."""
    radius = 84
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - probability)
    color = RISK_COLORS.get(risk, "#22D3EE")

    ui_html(
        f"""
        <div class="gauge-wrap">
        <svg width="210" height="210" viewBox="0 0 210 210">
        <circle cx="105" cy="105" r="{radius}" fill="none" stroke="rgba(148,163,184,0.13)" stroke-width="13"/>
        <circle class="gauge-arc" cx="105" cy="105" r="{radius}" fill="none" stroke="{color}" stroke-width="13"
        stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
        transform="rotate(-90 105 105)" opacity="0.95"/>
        <text class="gauge-value" x="105" y="100" text-anchor="middle" fill="{color}">{probability * 100:.0f}<tspan font-size="21">%</tspan></text>
        <text class="gauge-caption" x="105" y="126" text-anchor="middle" fill="#64748B">probabilidade</text>
        </svg>
        <div class="risk-chip" style="color:{color};background:{color}1A;border:1px solid {color}40;">
        <span class="dot" style="background:{color}"></span>risco {risk}
        </div>
        </div>
        """
    )


def render_skeleton() -> None:
    ui_html(
        """
        <div class="sk sk-circle"></div>
        <div class="sk sk-line" style="width:62%;margin-left:auto;margin-right:auto"></div>
        <div class="sk sk-line" style="width:84%"></div>
        <div class="sk sk-line" style="width:73%"></div>
        <div class="sk sk-line" style="width:79%"></div>
        """
    )


def render_telemetry() -> None:
    data = load_metrics()
    if not data:
        st.caption("Rode `python -m src.train` para gerar a telemetria do modelo.")
        return

    rows = "".join(
        f'<div class="metric-row"><span class="metric-name">{name}</span>'
        f'<span class="metric-val">{value:.2f}</span></div>'
        for name, value in (
            ("Acurácia", data["metrics"]["accuracy"]),
            ("Precision", data["metrics"]["precision"]),
            ("Recall", data["metrics"]["recall"]),
            ("F1-score", data["metrics"]["f1_score"]),
        )
    )
    ui_html(rows)

    st.markdown("")
    ui_html('<div class="card-title">Features mais importantes</div>')

    top = data.get("top_features", [])[:5]
    maximum = max((f["importance"] for f in top), default=1) or 1
    for feature in top:
        pct = feature["importance"] / maximum * 100
        ui_html(
            f"""
            <div style="margin-bottom:11px">
            <div style="display:flex;justify-content:space-between">
            <span class="feat-name">{feature['feature']}</span>
            <span class="feat-val">{feature['importance']:.3f}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
            </div>
            """
        )


# --------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------
online, latency, n_features = check_api()

ui_html(
    """
    <div class="hero">
    <div class="hero-badge"><span class="dot dot-ok"></span>Machine Learning · RAG · Function Calling</div>
    <h1>Churn Radar</h1>
    <p>Previsão de cancelamento de clientes de telecom com RandomForest servido por API REST, somada a um agente de IA que interpreta o resultado e recomenda ações de retenção.</p>
    </div>
    """
)

metrics_data = load_metrics()
ui_html(
    f"""
    <div class="hud">
    <div class="hud-item"><span class="dot {'dot-ok' if online else 'dot-off'}"></span>
    <div><span class="hud-label">API</span><span class="hud-value">{'online' if online else 'offline'}</span></div></div>
    <div class="hud-item"><div><span class="hud-label">Latência</span>
    <span class="hud-value">{f'{latency:.0f} ms' if latency else '—'}</span></div></div>
    <div class="hud-item"><div><span class="hud-label">Features</span>
    <span class="hud-value">{n_features or metrics_data.get('n_features', '—')}</span></div></div>
    <div class="hud-item"><div><span class="hud-label">Recall</span>
    <span class="hud-value">{metrics_data.get('metrics', {}).get('recall', '—')}</span></div></div>
    <div class="hud-item"><div><span class="hud-label">Amostras</span>
    <span class="hud-value">{metrics_data.get('n_samples', '—')}</span></div></div>
    </div>
    """
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    ui_html('<div class="card-title">Telemetria do modelo</div>')
    ui_html('<div class="card-sub">RandomForest · 100 árvores · depth 10</div>')
    render_telemetry()

    st.divider()
    ui_html('<div class="card-title">Conexão</div>')
    st.caption(f"`{API_URL}`")
    if not online:
        st.warning(
            "API offline. Suba com `uvicorn src.api:app --reload` ou aponte "
            "`CHURN_API_URL` para o serviço publicado.",
            icon="⚠️",
        )

tab_dashboard, tab_agent = st.tabs(["  Dashboard de Previsão  ", "  Assistente IA  "])

# --------------------------------------------------------------------------
# Aba 1 — Dashboard
# --------------------------------------------------------------------------
with tab_dashboard:
    col_form, col_result = st.columns([5, 6], gap="large")

    with col_form:
        ui_html('<div class="card-title">Perfil do cliente</div>')
        ui_html('<div class="card-sub">Os campos abaixo são os mesmos que a API valida — opções fora do domínio do treino são rejeitadas.</div>')

        with st.form("customer_form"):
            ui_html('<div class="form-section">Perfil</div>')
            c1, c2 = st.columns(2)
            gender = c1.selectbox(
                "Gênero", CATEGORY_OPTIONS["gender"], format_func=pt("gender")
            )
            senior = c2.selectbox(
                "Idoso", [0, 1], format_func=lambda v: "Sim" if v else "Não"
            )
            partner = c1.selectbox(
                "Cônjuge", CATEGORY_OPTIONS["Partner"], format_func=pt("Partner")
            )
            dependents = c2.selectbox(
                "Dependentes",
                CATEGORY_OPTIONS["Dependents"],
                format_func=pt("Dependents"),
            )

            ui_html('<div class="form-section">Conta</div>')
            tenure = st.slider("Tempo de casa (meses)", 0, 72, 5)
            c3, c4 = st.columns(2)
            monthly = c3.number_input("Mensalidade (R$)", 0.0, 1000.0, 85.5, step=5.0)
            total = c4.number_input("Total gasto (R$)", 0.0, 100_000.0, 450.75, step=50.0)
            contract = st.selectbox(
                "Contrato", CATEGORY_OPTIONS["Contract"], format_func=pt("Contract")
            )
            payment = st.selectbox(
                "Forma de pagamento",
                CATEGORY_OPTIONS["PaymentMethod"],
                format_func=pt("PaymentMethod"),
            )
            paperless = st.selectbox(
                "Fatura digital",
                CATEGORY_OPTIONS["PaperlessBilling"],
                format_func=pt("PaperlessBilling"),
            )

            ui_html('<div class="form-section">Serviços</div>')
            internet = st.selectbox(
                "Internet",
                CATEGORY_OPTIONS["InternetService"],
                format_func=pt("InternetService"),
            )
            c5, c6 = st.columns(2)
            phone = c5.selectbox(
                "Telefone", CATEGORY_OPTIONS["PhoneService"], format_func=pt("PhoneService")
            )
            multiple = c6.selectbox(
                "Múltiplas linhas",
                CATEGORY_OPTIONS["MultipleLines"],
                format_func=pt("MultipleLines"),
            )
            security = c5.selectbox(
                "Segurança online",
                CATEGORY_OPTIONS["OnlineSecurity"],
                format_func=pt("OnlineSecurity"),
            )
            backup = c6.selectbox(
                "Backup online",
                CATEGORY_OPTIONS["OnlineBackup"],
                format_func=pt("OnlineBackup"),
            )
            protection = c5.selectbox(
                "Proteção de aparelho",
                CATEGORY_OPTIONS["DeviceProtection"],
                format_func=pt("DeviceProtection"),
            )
            support = c6.selectbox(
                "Suporte técnico",
                CATEGORY_OPTIONS["TechSupport"],
                format_func=pt("TechSupport"),
            )
            tv = c5.selectbox(
                "Streaming de TV",
                CATEGORY_OPTIONS["StreamingTV"],
                format_func=pt("StreamingTV"),
            )
            movies = c6.selectbox(
                "Streaming de filmes",
                CATEGORY_OPTIONS["StreamingMovies"],
                format_func=pt("StreamingMovies"),
            )

            submitted = st.form_submit_button(
                "Analisar risco de churn", width="stretch"
            )

        if submitted:
            st.session_state.payload = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
            st.session_state.pending = True

    with col_result:
        slot = st.empty()

        if st.session_state.get("pending"):
            with slot.container():
                ui_html('<div class="card-title">Analisando</div>')
                ui_html('<div class="card-sub">Consultando o modelo via API…</div>')
                render_skeleton()

            result, error = call_predict(st.session_state.payload)
            st.session_state.pending = False
            st.session_state.result = result
            st.session_state.error = error
            slot.empty()

            if error:
                st.toast("Falha na análise", icon="⚠️")
            else:
                st.toast(f"Risco {result['risk_level'].lower()} identificado", icon="✅")

        result = st.session_state.get("result")
        error = st.session_state.get("error")

        if error:
            st.error(error, icon="🔌")
        elif result:
            with slot.container():
                ui_html('<div class="card-title">Resultado da análise</div>')
                ui_html('<div class="card-sub">Probabilidade estimada de cancelamento</div>')
                render_gauge(result["churn_probability"], result["risk_level"])

                st.markdown("")
                ui_html('<div class="card-title">Fatores observados no perfil</div>')
                drivers = risk_drivers(st.session_state.payload)
                ui_html(
                    "".join(
                        f'<div class="driver"><span class="driver-icon" '
                        f'style="color:{"#F87171" if kind == "risk" else "#34D399"}">'
                        f'{"▲" if kind == "risk" else "▼"}</span>'
                        f'<span class="driver-text">{text}</span></div>'
                        for kind, text in drivers
                    )
                    or '<p class="driver-text">Nenhum fator de destaque neste perfil.</p>'
                )
                st.caption(
                    "Leitura de perfil baseada nas features de maior importância "
                    "do modelo — não é atribuição por previsão (SHAP no roadmap)."
                )
        else:
            with slot.container():
                ui_html('<div class="card-title">Aguardando análise</div>')
                ui_html('<div class="card-sub">O radar entra em varredura quando um perfil é enviado</div>')
                render_radar(active=False)
                ui_html(
                    """
                    <div class="empty">
                    <h4>Nenhum cliente analisado ainda</h4>
                    <p>Preencha o perfil ao lado e envie para ver a probabilidade de churn e os fatores de risco.</p>
                    </div>
                    """
                )

# --------------------------------------------------------------------------
# Aba 2 — Assistente IA
# --------------------------------------------------------------------------
with tab_agent:
    ui_html('<div class="card-title">Assistente de retenção</div>')
    ui_html('<div class="card-sub">Agente com acesso ao modelo (function calling) e a uma base de conhecimento sobre churn (RAG)</div>')

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.agent_history = []

    if not st.session_state.messages:
        cols = st.columns(3, gap="medium")
        for col, (titulo, exemplo) in zip(
            cols,
            (
                ("Prever um cliente", "Cliente com 3 meses de casa, contrato mensal, fibra óptica, sem suporte técnico, mensalidade de 95. Qual o risco?"),
                ("Entender o conceito", "Quais são os principais fatores de risco de churn em telecom?"),
                ("Pedir recomendações", "Que ações de retenção funcionam melhor para clientes de contrato mensal?"),
            ),
        ):
            with col:
                ui_html(f'<div class="suggestion"><b>{titulo}</b>{exemplo}</div>')

        st.markdown("")
        ui_html(
            """
            <div class="empty">
            <h4>Comece uma conversa</h4>
            <p>O agente lembra do contexto: depois de analisar um cliente, você pode perguntar "e o que eu faço com ele?".</p>
            </div>
            """
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Pergunte sobre um cliente ou sobre churn…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consultando modelo e base de conhecimento…"):
                try:
                    from src.agent import chat as agent_chat

                    answer, st.session_state.agent_history = agent_chat(
                        prompt, st.session_state.agent_history
                    )
                except ImportError:
                    answer = (
                        "As dependências do agente não estão instaladas. "
                        "Rode `pip install -r requirements.txt`."
                    )
                except RuntimeError as exc:
                    answer = str(exc)
                except Exception as exc:
                    answer = f"O agente falhou: {exc}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()