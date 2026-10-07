"""Blocos visuais reaproveitados pelas páginas (cabeçalho, seções,
indicadores, notas, rodapé e o chip de status da API)."""

from __future__ import annotations

from collections.abc import Sequence

from src import __version__
from src.client import STATE_LOCAL_ONLY, STATE_OFFLINE, STATE_ONLINE, STATE_WAKING
from src.ui.data import get_client
from src.ui.html import esc, ui_html
from src.ui.settings import AUTHOR_NAME, GITHUB_URL, LINKEDIN_URL

# Ícones como SVG embutido (não dependem de uma fonte de ícones carregada):
# "call" e "hold" são do Material Symbols (Apache 2.0); GitHub e LinkedIn
# são as marcas oficiais, usadas só como link para os perfis.
ICONS = {
    "call": (
        "0 0 24 24",
        "M20.01 15.38c-1.23 0-2.42-.2-3.53-.56a.977.977 0 0 0-1.01.24l-1.57 1.97c-2.83-1.35-5.48-3.9-6.89-6.83l1.95-1.66c.27-.28.35-.67.24-1.02-.37-1.11-.56-2.3-.56-3.53 0-.54-.45-.99-.99-.99H4.19C3.65 3 3 3.24 3 3.99 3 13.28 10.73 21 20.01 21c.71 0 .99-.63.99-1.18v-3.45c0-.54-.45-.99-.99-.99z",
    ),
    "hold": (
        "0 0 24 24",
        "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zM7 11h10v2H7z",
    ),
    "github": (
        "0 0 16 16",
        "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z",
    ),
    "linkedin": (
        "0 0 24 24",
        "M20.45 20.45h-3.55v-5.57c0-1.33-.03-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28zM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12zM7.12 20.45H3.56V9h3.56v11.45zM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0z",
    ),
}


def icon(name: str, size: int = 18, color: str = "currentColor") -> str:
    view_box, path = ICONS[name]
    return (
        f'<svg width="{size}" height="{size}" viewBox="{view_box}" aria-hidden="true" '
        f'style="flex-shrink:0"><path fill="{color}" d="{path}"/></svg>'
    )


# ---------------------------------------------------------------------------
# Estado da API
# ---------------------------------------------------------------------------
_STATUS = {
    STATE_ONLINE: ("ok", "API online"),
    STATE_WAKING: ("warn", "API acordando · previsões locais"),
    STATE_OFFLINE: ("off", "API fora do ar · previsões locais"),
    STATE_LOCAL_ONLY: ("off", "Modo local"),
}

_STATUS_HELP = {
    STATE_ONLINE: "As previsões passam pela API FastAPI publicada no Render.",
    STATE_WAKING: (
        "O plano gratuito do Render desliga a API após 15 minutos sem uso, e ela leva cerca "
        "de 1 minuto para voltar. Enquanto isso, as previsões são calculadas aqui mesmo, com "
        "o mesmo modelo e o mesmo código."
    ),
    STATE_OFFLINE: (
        "A API não respondeu. As previsões são calculadas aqui mesmo, com o mesmo modelo e "
        "o mesmo código; uma nova tentativa acontece em 1 minuto."
    ),
    STATE_LOCAL_ONLY: "Nenhuma API configurada: as previsões são calculadas aqui mesmo.",
}


def status_chip() -> str:
    client = get_client()
    state = client.check()
    kind, label = _STATUS.get(state, ("off", "Verificando a API"))
    latency = client.status().get("latency_ms")
    if state == STATE_ONLINE and latency:
        label += f" · {latency:.0f} ms"
    return (
        f'<span class="cr-chip" title="{esc(_STATUS_HELP.get(state, ""))}">'
        f'<i class="cr-dot cr-dot--{kind}"></i>{esc(label)}</span>'
    )


def source_label(source: str, ms: float) -> str:
    if source == "api":
        return f"via API · {ms:.0f} ms"
    return f"cálculo local · {ms:.0f} ms"


# ---------------------------------------------------------------------------
# Estrutura das páginas
# ---------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, lead: str) -> None:
    ui_html(
        '<div class="cr-page-head">'
        '<div class="cr-page-head-main">'
        f'<p class="cr-eyebrow">{esc(eyebrow)}</p>'
        f'<div class="cr-h1" role="heading" aria-level="1">{esc(title)}</div>'
        f'<p class="cr-lead">{lead}</p>'
        "</div>"
        f'<div class="cr-page-head-side">{status_chip()}</div>'
        "</div>"
    )


def section(eyebrow: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="cr-sub">{sub}</p>' if sub else ""
    ui_html(
        f'<div class="cr-section"><p class="cr-eyebrow">{esc(eyebrow)}</p>'
        f'<div class="cr-h2" role="heading" aria-level="2">{esc(title)}</div>{sub_html}</div>'
    )


def kpis(items: Sequence[dict]) -> None:
    """items = [{"label", "value", "note", "accent"}] (note pode ter HTML)."""
    cards = "".join(
        f'<div class="cr-kpi{" cr-kpi--accent" if it.get("accent") else ""}">'
        f'<div class="cr-kpi-label">{esc(it["label"])}</div>'
        f'<div class="cr-kpi-value">{esc(it["value"])}</div>'
        + (f'<div class="cr-kpi-note">{it["note"]}</div>' if it.get("note") else "")
        + "</div>"
        for it in items
    )
    ui_html(f'<div class="cr-kpis">{cards}</div>')


def note(html: str, kind: str = "") -> None:
    ui_html(f'<div class="cr-note{" cr-note--" + kind if kind else ""}">{html}</div>')


def card(html: str, title: str = "") -> None:
    title_html = f'<p class="cr-card-title">{esc(title)}</p>' if title else ""
    ui_html(f'<div class="cr-card">{title_html}{html}</div>')


def footer() -> None:
    links = []
    if GITHUB_URL:
        links.append(
            f'<a href="{esc(GITHUB_URL)}" target="_blank" rel="noopener" '
            f'style="display:inline-flex;gap:6px;align-items:center">{icon("github", 15)}Código no GitHub</a>'
        )
    if LINKEDIN_URL:
        links.append(
            f'<a href="{esc(LINKEDIN_URL)}" target="_blank" rel="noopener" '
            f'style="display:inline-flex;gap:6px;align-items:center">{icon("linkedin", 14)}LinkedIn</a>'
        )
    links.append(
        '<a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" '
        'target="_blank" rel="noopener">Dados: IBM Telco Customer Churn</a>'
    )
    ui_html(
        '<div class="cr-footer">'
        f"<div>Churn Radar v{esc(__version__)} · projeto de {esc(AUTHOR_NAME)}</div>"
        f'<div class="cr-footer-links">{"".join(links)}</div>'
        "</div>"
    )
