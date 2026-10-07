"""Identidade visual: tokens de cor e o CSS da interface.

Fundo navy, ciano e índigo como acentos, Inter para texto e JetBrains Mono
para números e rótulos. As cores dos gráficos seguem regras fixas:

- faixas de risco usam as cores de status (verde, âmbar, vermelho) SEMPRE
  com rótulo e ícone — nunca só a cor;
- "aumenta o risco" e "reduz o risco" usam um par divergente laranja/azul,
  validado para daltonismo contra o fundo escuro (e não o vermelho/verde de
  status, que ficaria ambíguo com "risco alto/baixo");
- texto nunca usa a cor da série: valores e rótulos ficam nos tons de texto.

`STYLES` NÃO é f-string (ver src/ui/html.py).
"""

from __future__ import annotations

BG = "#05070D"
SURFACE = "#0B1120"
SURFACE_2 = "#0F172A"
LINE = "rgba(148, 163, 184, 0.16)"
TEXT = "#E6EDF7"
TEXT_2 = "#B8C4D6"
MUTED = "#8A98AE"
ACCENT = "#22D3EE"
ACCENT_2 = "#818CF8"
OK = "#34D399"
WARN = "#FBBF24"
DANGER = "#F87171"
UP = "#E36A36"  # aumenta o risco
DOWN = "#3A88E0"  # reduz o risco
NEUTRAL_BAR = "#475569"

RISK_STYLE = {
    "Baixo": {"color": OK, "icon": "▼", "css": "baixo"},
    "Médio": {"color": WARN, "icon": "◆", "css": "medio"},
    "Alto": {"color": DANGER, "icon": "▲", "css": "alto"},
}

STYLES = """
<style>
:root {
  --bg: #05070D;
  --surface: #0B1120;
  --surface-2: #0F172A;
  --surface-3: #131D33;
  --line: rgba(148, 163, 184, 0.16);
  --line-strong: rgba(148, 163, 184, 0.28);
  --accent: #22D3EE;
  --accent-2: #818CF8;
  --accent-soft: rgba(34, 211, 238, 0.10);
  --text: #E6EDF7;
  --text-2: #B8C4D6;
  --muted: #8A98AE;
  --ok: #34D399;
  --warn: #FBBF24;
  --danger: #F87171;
  --up: #E36A36;
  --down: #3A88E0;
  --neutral-bar: #475569;
  --sans: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
  --radius: 14px;
}

/* ---------- Estrutura ---------- */
.stApp { background: radial-gradient(1200px 520px at 12% -8%, rgba(34, 211, 238, 0.07), transparent 60%), radial-gradient(900px 480px at 100% 0%, rgba(129, 140, 248, 0.06), transparent 55%), var(--bg); }
[data-testid="stHeader"] { background: rgba(5, 7, 13, 0.82); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 4.6rem; padding-bottom: 2.5rem; }
/* Página Cliente: o formulário acompanha a rolagem dos resultados no desktop
   e, no celular, o resultado vem antes do formulário. O seletor desce só um
   nível (> stLayoutWrapper > stHorizontalBlock) para não afetar as colunas
   de dentro do formulário. */
@media (min-width: 761px) {
  .st-key-analysis > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { align-items: flex-start; }
  .st-key-analysis > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { position: sticky; top: 4.4rem; }
}
[data-testid="stTopNavLink"] { font-size: 14px; font-weight: 500; color: var(--text-2); border-radius: 9px; }
[data-testid="stTopNavLink"] p { font-size: 14px; }
[data-testid="stTopNavLink"][aria-current="page"] { background: var(--accent-soft); color: var(--text); box-shadow: inset 0 0 0 1px rgba(34, 211, 238, 0.28); }
[data-testid="stTopNavLink"]:hover { color: var(--text); }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* ---------- Tipografia base ---------- */
.cr-eyebrow { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 6px 0; }
.cr-h1 { font-size: 30px; line-height: 1.15; font-weight: 700; letter-spacing: -0.02em; color: var(--text); margin: 0 0 8px 0; }
.cr-lead { font-size: 15px; line-height: 1.6; color: var(--text-2); margin: 0; max-width: 72ch; }
.cr-h2 { font-size: 19px; line-height: 1.3; font-weight: 650; letter-spacing: -0.01em; color: var(--text); margin: 0 0 4px 0; }
.cr-sub { font-size: 13.5px; line-height: 1.55; color: var(--muted); margin: 0; max-width: 80ch; }
.cr-muted { color: var(--muted); }
.cr-mono { font-family: var(--mono); }
.cr-text { font-size: 14px; line-height: 1.65; color: var(--text-2); }
.cr-text b, .cr-text strong { color: var(--text); font-weight: 600; }
.cr-text code, .cr-note code, .cr-table code { font-family: var(--mono); font-size: 12px; color: #A5F3FC; background: rgba(34, 211, 238, 0.08); padding: 1px 6px; border-radius: 5px; }

/* ---------- Cabeçalho de página ---------- */
.cr-page-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 16px 24px; margin: 0 0 22px 0; }
.cr-page-head-main { flex: 1 1 520px; min-width: 0; }
.cr-page-head-side { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }

/* ---------- Chip de status da API ---------- */
.cr-chip { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); background: rgba(11, 17, 32, 0.85); border-radius: 999px; padding: 6px 12px; font-family: var(--mono); font-size: 12px; color: var(--text-2); white-space: nowrap; }
.cr-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; background: var(--muted); }
.cr-dot--ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.16); }
.cr-dot--warn { background: var(--warn); box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.16); animation: cr-pulse 1.6s ease-in-out infinite; }
.cr-dot--off { background: var(--muted); }
@keyframes cr-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ---------- Seções e cartões ---------- */
.cr-section { margin: 34px 0 14px 0; }
.cr-section:first-child { margin-top: 8px; }
.cr-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px 22px; }
.cr-card + .cr-card { margin-top: 14px; }
.cr-card-title { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0 0 12px 0; }
.st-key-panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 18px 20px 6px 20px; }
.cr-group { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 14px 0 4px 0; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
.cr-group:first-child { margin-top: 2px; }

/* ---------- Indicadores ---------- */
.cr-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.cr-kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 15px 16px 14px 16px; min-width: 0; }
.cr-kpi-label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.cr-kpi-value { font-family: var(--mono); font-size: 25px; font-weight: 650; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; }
.cr-kpi-note { font-size: 12.5px; line-height: 1.5; color: var(--text-2); margin-top: 7px; }
.cr-kpi--accent { border-top: 2px solid var(--accent); }

/* ---------- Veredito do cliente ---------- */
.cr-verdict { position: relative; overflow: hidden; background: linear-gradient(135deg, rgba(34, 211, 238, 0.08), rgba(129, 140, 248, 0.05) 55%, rgba(11, 17, 32, 0) 80%), var(--surface); border: 1px solid var(--line-strong); border-radius: var(--radius); padding: 22px 24px; }
.cr-verdict-top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px 20px; }
.cr-prob { font-family: var(--mono); font-size: 56px; font-weight: 700; letter-spacing: -0.04em; line-height: 0.95; color: var(--text); }
.cr-prob small { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; color: var(--text-2); }
.cr-prob-label { font-size: 13px; color: var(--text-2); margin-top: 8px; }
.cr-risk { display: inline-flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 13px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; padding: 7px 14px; border-radius: 999px; border: 1px solid; }
.cr-risk--baixo { color: var(--ok); border-color: rgba(52, 211, 153, 0.45); background: rgba(52, 211, 153, 0.10); }
.cr-risk--medio { color: var(--warn); border-color: rgba(251, 191, 36, 0.45); background: rgba(251, 191, 36, 0.10); }
.cr-risk--alto { color: var(--danger); border-color: rgba(248, 113, 113, 0.45); background: rgba(248, 113, 113, 0.10); }
.cr-reco { display: flex; gap: 12px; align-items: flex-start; margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line); }
.cr-reco-icon { flex-shrink: 0; width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-weight: 700; font-size: 15px; }
.cr-reco-icon--go { background: rgba(34, 211, 238, 0.14); color: var(--accent); }
.cr-reco-icon--hold { background: rgba(148, 163, 184, 0.12); color: var(--text-2); }
.cr-reco-title { font-size: 15px; font-weight: 600; color: var(--text); }
.cr-reco-text { font-size: 13.5px; line-height: 1.55; color: var(--text-2); margin-top: 2px; }
.cr-meta { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 14px; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.cr-meta b { color: var(--text-2); font-weight: 500; }

/* ---------- Legendas ---------- */
.cr-legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--text-2); margin: 2px 0 12px 0; }
.cr-legend span { display: inline-flex; align-items: center; gap: 7px; }
.cr-swatch { width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
.cr-swatch--ring { background: var(--surface) !important; border: 2px solid var(--text-2); border-radius: 50%; }
.cr-swatch--line { height: 3px; border-radius: 2px; width: 16px; }

/* ---------- Gráficos (HTML + SVG, responsivos) ---------- */
.cr-chart { position: relative; width: 100%; }
.cr-plot { position: relative; width: 100%; }
.cr-plot svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.cr-grid-line { position: absolute; left: 0; right: 0; height: 1px; background: var(--line); }
.cr-vline { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1px solid var(--line-strong); }
.cr-vline-label { position: absolute; top: -20px; transform: translateX(-50%); font-family: var(--mono); font-size: 11px; color: var(--text-2); white-space: nowrap; }
.cr-axis { position: relative; height: 22px; margin-top: 6px; }
.cr-axis span { position: absolute; top: 0; transform: translateX(-50%); font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }
.cr-axis span.cr-axis-start { transform: none; }
.cr-axis span.cr-axis-end { transform: translateX(-100%); }
.cr-yaxis span { position: absolute; right: 10px; transform: translateY(-50%); font-family: var(--mono); font-size: 11px; color: var(--muted); white-space: nowrap; }
.cr-axis-title { font-family: var(--mono); font-size: 11px; color: var(--muted); text-align: center; margin-top: 2px; }
.cr-hbars { display: flex; align-items: flex-end; gap: 2px; height: 100%; }
.cr-hbar { flex: 1; min-width: 0; height: 100%; display: flex; align-items: flex-end; position: relative; }
.cr-hbar > i { display: block; width: 100%; border-radius: 4px 4px 0 0; background: var(--neutral-bar); }
.cr-hbar:hover > i { filter: brightness(1.35); }
.cr-marker { position: absolute; width: 12px; height: 12px; border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--surface); }
.cr-hit { position: absolute; top: 0; bottom: 0; }
.cr-hit::after { content: ''; position: absolute; top: 0; bottom: 0; left: 50%; border-left: 1px solid var(--text-2); opacity: 0; }
.cr-hit:hover::after { opacity: 0.55; }
.cr-tip { display: none; position: absolute; z-index: 5; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%); background: #111A2E; border: 1px solid var(--line-strong); border-radius: 9px; padding: 8px 10px; font-family: var(--mono); font-size: 11.5px; line-height: 1.55; color: var(--text); white-space: nowrap; box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45); pointer-events: none; }
.cr-hit:hover .cr-tip, .cr-hbar:hover .cr-tip, .cr-row:hover .cr-tip { display: block; }
.cr-tip b { color: var(--accent); font-weight: 600; }
.cr-hit--left .cr-tip { left: 0; transform: none; }
.cr-hit--right .cr-tip { left: auto; right: 0; transform: none; }

/* Faixas de risco sob a escala */
.cr-zones { position: relative; height: 26px; margin-top: 6px; border-radius: 7px; overflow: hidden; display: flex; }
.cr-zone { height: 100%; display: flex; align-items: center; justify-content: center; font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: 0.04em; white-space: nowrap; overflow: hidden; }
.cr-zone--baixo { background: rgba(52, 211, 153, 0.13); color: var(--ok); }
.cr-zone--medio { background: rgba(251, 191, 36, 0.13); color: var(--warn); border-left: 2px solid var(--surface); }
.cr-zone--alto { background: rgba(248, 113, 113, 0.13); color: var(--danger); border-left: 2px solid var(--surface); }

/* Linhas com trilha (contribuições, cenários, PSI, coeficientes) */
.cr-rows { display: flex; flex-direction: column; gap: 4px; }
.cr-row { position: relative; display: grid; grid-template-columns: minmax(140px, 30%) 1fr 84px; align-items: center; gap: 14px; padding: 7px 8px; border-radius: 9px; }
.cr-row:hover { background: rgba(148, 163, 184, 0.06); }
.cr-row-label { font-size: 13.5px; color: var(--text); line-height: 1.35; min-width: 0; }
.cr-row-label small { display: block; font-size: 12px; color: var(--muted); margin-top: 1px; }
.cr-row-value { font-family: var(--mono); font-size: 13px; color: var(--text); text-align: right; white-space: nowrap; }
.cr-row-value small { display: block; font-size: 11px; color: var(--muted); }
.cr-row--total .cr-row-label { font-weight: 600; }
.cr-row--axis { padding-top: 0; }
.cr-row--axis:hover { background: transparent; }
.cr-row--total { border-top: 1px solid var(--line); border-radius: 0; margin-top: 4px; padding-top: 10px; }
.cr-track { position: relative; height: 14px; border-radius: 4px; background: rgba(148, 163, 184, 0.08); }
.cr-bar { position: absolute; top: 0; bottom: 0; border-radius: 4px; min-width: 3px; }
.cr-bar--up { background: var(--up); }
.cr-bar--down { background: var(--down); }
.cr-bar--neutral { background: var(--accent); }
.cr-tick { position: absolute; top: -3px; bottom: -3px; width: 2px; border-radius: 1px; background: var(--text-2); }
.cr-center { position: absolute; top: -4px; bottom: -4px; left: 50%; width: 1px; background: var(--line-strong); }
.cr-dumbbell-line { position: absolute; top: 50%; height: 2px; transform: translateY(-50%); }
.cr-dumbbell-dot { position: absolute; top: 50%; width: 12px; height: 12px; border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--surface); }
.cr-dumbbell-dot--now { background: var(--surface); border: 2px solid var(--text-2); }
.cr-tag { display: inline-block; font-family: var(--mono); font-size: 11px; font-weight: 500; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line-strong); color: var(--text-2); margin-top: 4px; }
.cr-tag--ok { color: var(--ok); border-color: rgba(52, 211, 153, 0.4); }
.cr-tag--warn { color: var(--warn); border-color: rgba(251, 191, 36, 0.4); }
.cr-tag--danger { color: var(--danger); border-color: rgba(248, 113, 113, 0.4); }
.cr-tag--accent { color: var(--accent); border-color: rgba(34, 211, 238, 0.4); }

/* ---------- Notas ---------- */
.cr-note { background: var(--surface-2); border: 1px solid var(--line); border-left: 3px solid var(--accent-2); border-radius: 10px; padding: 12px 14px; font-size: 13px; line-height: 1.6; color: var(--text-2); margin: 12px 0 0 0; }
.cr-note b { color: var(--text); font-weight: 600; }
.cr-note--warn { border-left-color: var(--warn); }
.cr-note--ok { border-left-color: var(--ok); }

/* ---------- Tabelas ---------- */
.cr-table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.cr-table { width: 100%; border-collapse: collapse; min-width: 560px; }
.cr-table th, .cr-table td { border: none !important; background: transparent !important; }
.cr-table th { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); text-align: right; padding: 11px 14px; border-bottom: 1px solid var(--line) !important; white-space: nowrap; }
.cr-table td { font-family: var(--mono); font-size: 13px; color: var(--text-2); text-align: right; padding: 10px 14px; border-bottom: 1px solid rgba(148, 163, 184, 0.08) !important; white-space: nowrap; }
.cr-table th:first-child, .cr-table td:first-child { text-align: left; }
.cr-table td:first-child { font-family: var(--sans); font-size: 13.5px; color: var(--text); }
.cr-table tr:last-child td { border-bottom: none !important; }
.cr-table tr.cr-hl td { background: rgba(34, 211, 238, 0.06) !important; color: var(--text); }

/* ---------- Página inicial ---------- */
.cr-hero { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 20px; padding: 40px 40px 34px 40px; background: radial-gradient(120% 160% at 10% 0%, rgba(34, 211, 238, 0.14), rgba(129, 140, 248, 0.07) 40%, rgba(5, 7, 13, 0) 72%), var(--surface); }
.cr-hero::after { content: ''; position: absolute; inset: 0; background-image: linear-gradient(rgba(34, 211, 238, 0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(34, 211, 238, 0.05) 1px, transparent 1px); background-size: 36px 36px; -webkit-mask-image: radial-gradient(70% 100% at 20% 0%, #000, transparent 75%); mask-image: radial-gradient(70% 100% at 20% 0%, #000, transparent 75%); pointer-events: none; }
.cr-hero > * { position: relative; z-index: 1; }
.cr-hero-title { font-size: 44px; line-height: 1.05; font-weight: 750; letter-spacing: -0.03em; margin: 0 0 14px 0; background: linear-gradient(96deg, #F8FAFC 10%, var(--accent) 55%, var(--accent-2) 95%); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.cr-hero-text { font-size: 17px; line-height: 1.6; color: var(--text-2); max-width: 64ch; margin: 0; }
.cr-stack { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
.cr-stack span { font-family: var(--mono); font-size: 12px; color: var(--text-2); border: 1px solid var(--line); background: rgba(5, 7, 13, 0.55); border-radius: 999px; padding: 5px 11px; }
.cr-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; counter-reset: step; }
.cr-step { position: relative; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px 16px 15px 16px; }
.cr-step-n { font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; color: var(--accent); text-transform: uppercase; }
.cr-step-t { font-size: 15px; font-weight: 600; color: var(--text); margin: 6px 0 6px 0; }
.cr-step-d { font-size: 13px; line-height: 1.55; color: var(--text-2); }
.cr-step-d code { font-family: var(--mono); font-size: 11.5px; color: #A5F3FC; }
.cr-decisions { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.cr-decisions .cr-card, .cr-steps .cr-card { margin-top: 0 !important; }
.st-key-cta > [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] { background: linear-gradient(96deg, #22D3EE, #67E8F9); border-color: transparent; }
.st-key-cta > [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] p, .st-key-cta > [data-testid="stElementContainer"]:first-child [data-testid="stPageLink-NavLink"] span { color: #03131A !important; font-weight: 650; }

/* ---------- Assistente ---------- */
.cr-trace { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 8px 0; }
.cr-trace span { font-family: var(--mono); font-size: 11.5px; color: var(--text-2); border: 1px solid var(--line); background: var(--surface-2); border-radius: 999px; padding: 3px 10px; }
.cr-trace span b { color: var(--accent); font-weight: 600; }
.cr-recorded { font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent-2); margin-bottom: 6px; }
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p, [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { font-size: 14.5px; line-height: 1.65; color: var(--text-2); }
[data-testid="stChatMessage"] strong { color: var(--text); }

/* ---------- Rodapé ---------- */
.cr-footer { margin-top: 56px; padding: 22px 0 6px 0; border-top: 1px solid var(--line); display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px 24px; font-size: 13px; color: var(--muted); }
.cr-footer a { color: var(--text-2) !important; text-decoration: none; }
.cr-footer a:hover { color: var(--accent) !important; }
.cr-footer-links { display: flex; flex-wrap: wrap; gap: 6px 18px; }

/* ---------- Componentes do Streamlit ---------- */
[data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 500 !important; color: var(--text-2) !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: 12px !important; background: var(--surface); }
[data-testid="stExpander"] summary p { font-size: 14px; font-weight: 500; color: var(--text-2); }
[data-testid="stDataFrame"] { border-radius: 12px; }
[data-testid="stBaseButton-primary"] { background: linear-gradient(96deg, #22D3EE, #67E8F9) !important; color: #03131A !important; border: none !important; font-weight: 650 !important; }
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.06); box-shadow: 0 0 22px rgba(34, 211, 238, 0.28); }
[data-testid="stBaseButton-secondary"] { background: var(--surface-2) !important; }
[data-testid="stPageLink-NavLink"] { border: 1px solid var(--line); background: var(--surface-2); border-radius: 10px; padding: 6px 12px; }
[data-testid="stPageLink-NavLink"]:hover { border-color: rgba(34, 211, 238, 0.45); }

/* ---------- Responsivo ---------- */
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-top: 4.2rem; }
  .cr-h1 { font-size: 25px; }
  .cr-hero { padding: 26px 20px 22px 20px; border-radius: 16px; }
  .cr-hero-title { font-size: 33px; }
  .cr-hero-text { font-size: 15px; }
  .cr-prob { font-size: 46px; }
  .cr-row { grid-template-columns: 1fr 78px; gap: 6px 10px; }
  .cr-row .cr-track { grid-column: 1 / -1; grid-row: 2; }
  .cr-row--axis > div:first-child, .cr-row--axis > div:last-child { display: none; }
  .cr-row--axis > div:nth-child(2) { grid-column: 1 / -1; }
  .cr-card, .cr-verdict { padding: 17px 16px; }
  .st-key-analysis > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) { order: -1; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""