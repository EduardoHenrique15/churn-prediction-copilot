"""Gráficos feitos à mão em HTML + SVG — funções puras que devolvem markup.

Por que não uma biblioteca de gráficos: o texto dos gráficos precisa ficar
legível no celular. Um SVG com viewBox fixo encolhe o texto junto com o
desenho (11 px viram 6 px numa tela de 360 px). Aqui o desenho é que escala:
barras e pontos são HTML posicionado em %, linhas são SVG com
`preserveAspectRatio="none"` e traço que não escala, e todo texto é HTML.

Regras seguidas em todos os gráficos: barras de até 24 px com cantos de
4 px, linhas de 2 px, marcadores de 12 px com anel da cor do fundo, grade
sólida e discreta, legenda sempre que houver duas séries, texto sempre nos
tons de texto (nunca na cor da série) e dica ao passar o mouse.

Nada aqui chama o Streamlit: quem renderiza é `ui_html()`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from src.business import fmt_int, fmt_pct, fmt_pp
from src.ui.html import esc
from src.ui.theme import ACCENT, DANGER, DOWN, MUTED, NEUTRAL_BAR, OK, RISK_STYLE, TEXT_2, UP, WARN


def _pos(value: float) -> str:
    """Posição CSS em % (sempre com ponto decimal)."""
    return f"{max(0.0, min(100.0, value)):.3f}%"


def legend(items: Sequence[tuple[str, str, str]]) -> str:
    """items = (rótulo, cor, tipo) com tipo 'box', 'ring' ou 'line'."""
    parts = []
    for label, color, kind in items:
        css = {"ring": "cr-swatch cr-swatch--ring", "line": "cr-swatch cr-swatch--line"}.get(
            kind, "cr-swatch"
        )
        style = f"border-color:{color}" if kind == "ring" else f"background:{color}"
        parts.append(f'<span><i class="{css}" style="{style}"></i>{esc(label)}</span>')
    return '<div class="cr-legend">' + "".join(parts) + "</div>"


def _num(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _axis(ticks: Sequence[tuple[float, str]]) -> str:
    """Eixo horizontal: ticks = (posição 0-100, rótulo). Os rótulos das pontas
    se alinham para dentro, para não vazar do gráfico."""
    spans = []
    for x, t in ticks:
        cls = (
            ' class="cr-axis-start"' if x <= 0.5 else (' class="cr-axis-end"' if x >= 99.5 else "")
        )
        spans.append(f'<span{cls} style="left:{_pos(x)}">{esc(t)}</span>')
    return f'<div class="cr-axis" aria-hidden="true">{"".join(spans)}</div>'


PCT_TICKS = [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]


# ---------------------------------------------------------------------------
# Escala de risco com a distribuição da base
# ---------------------------------------------------------------------------
def risk_scale(
    probabilities: Sequence[float],
    cuts: dict,
    marker: float | None = None,
    marker_label: str = "Este cliente",
    threshold_label: str | None = None,
    threshold: float | None = None,
    height: int = 110,
    population: str = "clientes",
) -> str:
    """Histograma (faixas de 5%) + faixas de risco + marcador opcional."""
    probs = np.asarray(probabilities, dtype=float)
    counts, edges = np.histogram(probs, bins=np.linspace(0, 1, 21))
    peak = max(int(counts.max()), 1)
    total = max(len(probs), 1)
    marker_bin = None if marker is None else min(int(marker * 20), 19)

    bars = []
    for i, n in enumerate(counts):
        h = max(n / peak * 100, 1.5 if n else 0)
        color = ACCENT if i == marker_bin else NEUTRAL_BAR
        tip = (
            f'<span class="cr-tip"><b>{fmt_pct(edges[i], 0)} a {fmt_pct(edges[i + 1], 0)}</b>'
            f"<br>{fmt_int(n)} {esc(population)} ({fmt_pct(n / total)})</span>"
        )
        bars.append(
            f'<div class="cr-hbar"><i style="height:{h:.2f}%;background:{color}"></i>{tip}</div>'
        )

    overlays = []
    if threshold is not None:
        overlays.append(
            f'<div class="cr-vline" style="left:{_pos(threshold * 100)};border-left:1px solid {TEXT_2}">'
            f'<span class="cr-vline-label">{esc(threshold_label or "corte")}</span></div>'
        )
    if marker is not None:
        overlays.append(
            f'<div class="cr-vline" style="left:{_pos(marker * 100)};border-left:2px solid {ACCENT}">'
            f'<span class="cr-vline-label" style="color:#E6EDF7;font-weight:600">'
            f"{esc(marker_label)} · {fmt_pct(marker)}</span></div>"
        )

    low, mid = cuts["baixo_max"], cuts["medio_max"]
    zones = [
        ("baixo", 0.0, low, "Baixo"),
        ("medio", low, mid, "Médio"),
        ("alto", mid, 1.0, "Alto"),
    ]
    zone_html = "".join(
        f'<div class="cr-zone cr-zone--{css}" style="width:{(b - a) * 100:.3f}%" '
        f'title="Risco {label.lower()}: {fmt_pct(a, 0)} a {fmt_pct(b, 0)}">'
        f"{RISK_STYLE[label]['icon']} {label}</div>"
        for css, a, b, label in zones
        if b > a
    )
    summary = f"Distribuição de {fmt_int(total)} {population} por probabilidade de cancelar" + (
        f"; marcador em {fmt_pct(marker)}" if marker is not None else ""
    )
    return (
        f'<div class="cr-chart" role="img" aria-label="{esc(summary)}" style="padding-top:22px">'
        f'<div class="cr-plot" style="height:{height}px">'
        f'<div class="cr-hbars">{"".join(bars)}</div>{"".join(overlays)}</div>'
        f'<div class="cr-zones">{zone_html}</div>{_axis(PCT_TICKS)}</div>'
    )


# ---------------------------------------------------------------------------
# Linhas com trilha: contribuições, cenários, PSI, coeficientes
# ---------------------------------------------------------------------------
def _row(
    label: str,
    track: str,
    value: str,
    sub: str = "",
    value_sub: str = "",
    extra_cls: str = "",
    title: str = "",
) -> str:
    sub_html = f"<small>{sub}</small>" if sub else ""
    value_sub_html = f"<small>{value_sub}</small>" if value_sub else ""
    title_attr = f' title="{esc(title)}"' if title else ""
    return (
        f'<div class="cr-row {extra_cls}"{title_attr}>'
        f'<div class="cr-row-label">{label}{sub_html}</div>'
        f'<div class="cr-track">{track}</div>'
        f'<div class="cr-row-value">{value}{value_sub_html}</div></div>'
    )


def waterfall(base_p: float, steps: Sequence[dict], final_p: float) -> str:
    """Do ponto de partida até a previsão do cliente, uma característica por vez.

    steps = [{"label", "value", "p_start", "p_end"}], já em ordem.
    """
    rows = [
        _row(
            "Ponto de partida",
            f'<i class="cr-tick" style="left:{_pos(base_p * 100)}"></i>',
            fmt_pct(base_p, 0),
            sub="cliente médio da base",
            title="Previsão do modelo para um cliente com cada característica na média da base.",
        )
    ]
    for step in steps:
        a, b = step["p_start"], step["p_end"]
        up = b >= a
        left, width = min(a, b) * 100, abs(b - a) * 100
        rows.append(
            _row(
                esc(step["label"]),
                f'<i class="cr-bar cr-bar--{"up" if up else "down"}" '
                f'style="left:{_pos(left)};width:{max(width, 0.4):.3f}%"></i>',
                fmt_pp(b - a),
                sub=esc(step["value"]),
                title=f"{step['label']}: {step['value']} — leva o risco de {fmt_pct(a)} para {fmt_pct(b)}",
            )
        )
    rows.append(
        _row(
            "Este cliente",
            f'<i class="cr-tick" style="left:{_pos(final_p * 100)};background:{ACCENT}"></i>',
            fmt_pct(final_p),
            extra_cls="cr-row--total",
        )
    )
    axis = (
        '<div class="cr-row cr-row--axis"><div></div><div>'
        + _axis(PCT_TICKS)
        + "</div><div></div></div>"
    )
    return (
        legend([("Aumenta o risco", UP, "box"), ("Reduz o risco", DOWN, "box")])
        + '<div class="cr-rows" role="list">'
        + "".join(rows)
        + axis
        + "</div>"
    )


def whatif(current_p: float, scenarios: Sequence[dict]) -> str:
    """Halteres: hoje (anel) → com a mudança (ponto cheio).

    scenarios = [{"label", "p", "tag", "tag_kind"}], já ordenados.
    """
    rows = []
    for sc in scenarios:
        p = sc["p"]
        delta = p - current_p
        color = DOWN if delta < 0 else UP
        left, width = min(p, current_p) * 100, abs(delta) * 100
        track = (
            f'<i class="cr-dumbbell-line" style="left:{_pos(left)};width:{width:.3f}%;background:{color}"></i>'
            f'<i class="cr-dumbbell-dot cr-dumbbell-dot--now" style="left:{_pos(current_p * 100)}"></i>'
            f'<i class="cr-dumbbell-dot" style="left:{_pos(p * 100)};background:{color}"></i>'
        )
        tag = (
            f'<span class="cr-tag cr-tag--{sc.get("tag_kind", "accent")}">{esc(sc["tag"])}</span>'
            if sc.get("tag")
            else ""
        )
        rows.append(
            _row(
                esc(sc["label"]) + (f"<br>{tag}" if tag else ""),
                track,
                fmt_pct(p),
                value_sub=fmt_pp(delta),
                title=f"{sc['label']}: {fmt_pct(current_p)} → {fmt_pct(p)}",
            )
        )
    axis = (
        '<div class="cr-row cr-row--axis"><div></div><div>'
        + _axis(PCT_TICKS)
        + "</div><div></div></div>"
    )
    return (
        legend(
            [
                ("Hoje", TEXT_2, "ring"),
                ("Reduz o risco", DOWN, "box"),
                ("Aumenta o risco", UP, "box"),
            ]
        )
        + '<div class="cr-rows">'
        + "".join(rows)
        + axis
        + "</div>"
    )


def diverging_bars(rows: Sequence[dict], limit: float, fmt: Callable[[float], str]) -> str:
    """Barras a partir do centro: rows = [{"label", "value", "sub"}] com
    value em escala simétrica [-limit, +limit] (positivo = aumenta o risco)."""
    out = []
    for r in rows:
        v = max(-limit, min(limit, r["value"]))
        half = abs(v) / limit * 50
        left = 50 if v >= 0 else 50 - half
        cls = "up" if v >= 0 else "down"
        track = (
            '<i class="cr-center"></i>'
            f'<i class="cr-bar cr-bar--{cls}" style="left:{_pos(left)};width:{half:.3f}%"></i>'
        )
        out.append(_row(esc(r["label"]), track, fmt(r["value"]), sub=esc(r.get("sub", ""))))
    return (
        legend(
            [
                ("Aumenta a chance de cancelar", UP, "box"),
                ("Reduz a chance de cancelar", DOWN, "box"),
            ]
        )
        + '<div class="cr-rows">'
        + "".join(out)
        + "</div>"
    )


def psi_bars(rows: Sequence[dict], stable: float, shift: float) -> str:
    """PSI por variável, com as linhas de corte de 'atenção' e 'mudança forte'."""
    # Escala limitada a 1,0: um PSI de 5 esmagaria as linhas de 0,10 e 0,25
    # contra a borda. Barras acima do limite ficam cheias e o valor aparece.
    top = min(max([r["psi"] for r in rows] + [shift * 1.6]), 1.0)
    colors = {"Estável": (OK, "ok"), "Atenção": (WARN, "warn"), "Mudança forte": (DANGER, "danger")}
    out = []
    for r in rows:
        color, kind = colors[r["status"]]
        width = min(r["psi"] / top, 1.0) * 100
        track = (
            f'<i class="cr-bar" style="left:0;width:{max(width, 0.6):.3f}%;background:{color}"></i>'
            f'<i class="cr-tick" style="left:{_pos(stable / top * 100)};width:1px;background:{MUTED}"></i>'
            f'<i class="cr-tick" style="left:{_pos(shift / top * 100)};width:1px;background:{MUTED}"></i>'
        )
        status = f'<span class="cr-tag cr-tag--{kind}">{esc(r["status"])}</span>'
        out.append(
            _row(
                esc(r["label"]) + f"<br>{status}",
                track,
                _num(r["psi"], 3),
                title=f"PSI {r['psi']:.3f} — {r['status']}",
            )
        )
    note = (
        '<div class="cr-row cr-row--axis"><div></div><div>'
        + _axis([(0, "0"), (stable / top * 100, _num(stable)), (shift / top * 100, _num(shift))])
        + "</div><div></div></div>"
    )
    return '<div class="cr-rows">' + "".join(out) + note + "</div>"


def stacked_bar(parts: Sequence[dict]) -> str:
    """Uma barra 100% com segmentos rotulados: parts = [{"label", "n", "color"}]."""
    total = max(sum(p["n"] for p in parts), 1)
    segments = "".join(
        f'<div style="width:{p["n"] / total * 100:.3f}%;background:{p["color"]};height:100%;'
        f'border-radius:4px" title="{esc(p["label"])}: {fmt_int(p["n"])} ({fmt_pct(p["n"] / total)})"></div>'
        for p in parts
        if p["n"]
    )
    labels = "".join(
        f'<span><i class="cr-swatch" style="background:{p["color"]}"></i>{esc(p["label"])}: '
        f'<b class="cr-mono" style="color:#E6EDF7;font-weight:600">{fmt_int(p["n"])}</b> '
        f"({fmt_pct(p['n'] / total)})</span>"
        for p in parts
    )
    return (
        f'<div style="display:flex;gap:2px;height:22px">{segments}</div>'
        f'<div class="cr-legend" style="margin-top:10px">{labels}</div>'
    )


# ---------------------------------------------------------------------------
# Gráfico de linhas (valor × corte, ganho acumulado, dependência parcial...)
# ---------------------------------------------------------------------------
def line_chart(
    series: Sequence[dict],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    x_ticks: Sequence[tuple[float, str]],
    y_ticks: Sequence[tuple[float, str]],
    height: int = 260,
    hover: Sequence[tuple[float, str]] = (),
    markers: Sequence[dict] = (),
    vlines: Sequence[dict] = (),
    x_title: str = "",
    aria: str = "",
) -> str:
    """series = [{"points": [(x, y)], "color", "width", "fill", "dash"}].

    `hover` = (x, html) — uma área de passar o mouse por ponto, com linha
    vertical e dica. `markers` = [{"x", "y", "color", "label"}]. `vlines` =
    [{"x", "label", "color"}].
    """
    x0, x1 = x_domain
    y0, y1 = y_domain

    def nx(x: float) -> float:
        return (x - x0) / (x1 - x0) * 100

    def ny(y: float) -> float:
        return (1 - (y - y0) / (y1 - y0)) * 100

    grid = "".join(
        f'<div class="cr-grid-line" style="top:{_pos(ny(v))}"></div>' for v, _ in y_ticks
    )
    paths = []
    for s in series:
        pts = [(nx(x) * 10, ny(y) * 10) for x, y in s["points"]]
        coords = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
        if s.get("fill") and pts:
            base = ny(max(y0, min(y1, 0))) * 10
            area = f"{pts[0][0]:.2f},{base:.2f} {coords} {pts[-1][0]:.2f},{base:.2f}"
            paths.append(
                f'<polygon points="{area}" fill="{s["color"]}" fill-opacity="0.10" stroke="none"/>'
            )
        dash = ' stroke-dasharray="6 5"' if s.get("dash") else ""
        paths.append(
            f'<polyline points="{coords}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="{s.get("width", 2)}" stroke-linejoin="round" stroke-linecap="round" '
            f'vector-effect="non-scaling-stroke"{dash}/>'
        )
    svg = (
        '<svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">'
        + "".join(paths)
        + "</svg>"
    )
    lines = "".join(
        f'<div class="cr-vline" style="left:{_pos(nx(v["x"]))};border-left:1px solid {v.get("color", MUTED)}">'
        f'<span class="cr-vline-label">{esc(v["label"])}</span></div>'
        for v in vlines
    )
    dots = "".join(
        f'<i class="cr-marker" style="left:{_pos(nx(m["x"]))};top:{_pos(ny(m["y"]))};background:{m["color"]}" '
        f'title="{esc(m.get("label", ""))}"></i>'
        for m in markers
    )
    hits = []
    if hover:
        xs = [h[0] for h in hover]
        step = (max(xs) - min(xs)) / max(len(xs) - 1, 1)
        w = step / (x1 - x0) * 100
        for i, (x, html) in enumerate(hover):
            side = (
                " cr-hit--left"
                if i < len(hover) * 0.15
                else (" cr-hit--right" if i > len(hover) * 0.85 else "")
            )
            hits.append(
                f'<div class="cr-hit{side}" style="left:{_pos(nx(x) - w / 2)};width:{w:.3f}%">'
                f'<span class="cr-tip">{html}</span></div>'
            )
    y_labels = "".join(f'<span style="top:{_pos(ny(v))}">{esc(t)}</span>' for v, t in y_ticks)
    x_title_html = f'<div class="cr-axis-title">{esc(x_title)}</div>' if x_title else ""
    return (
        f'<div class="cr-chart" role="img" aria-label="{esc(aria)}" '
        f'style="display:grid;grid-template-columns:76px 1fr;padding-top:24px">'
        f'<div class="cr-yaxis" style="position:relative;height:{height}px">{y_labels}</div>'
        f'<div><div class="cr-plot" style="height:{height}px">{grid}{svg}{lines}{dots}{"".join(hits)}</div>'
        f"{_axis([(nx(v), t) for v, t in x_ticks])}{x_title_html}</div></div>"
    )


def dot_ranges(
    rows: Sequence[dict],
    domain: tuple[float, float],
    floor: float | None,
    fmt: Callable[[float], str],
) -> str:
    """Média ± erro-padrão por candidato: rows = [{"label", "mean", "se", "chosen", "sub"}]."""
    d0, d1 = domain

    def nx(v: float) -> float:
        return (v - d0) / (d1 - d0) * 100

    out = []
    for r in rows:
        lo, hi = r["mean"] - r["se"], r["mean"] + r["se"]
        color = ACCENT if r.get("chosen") else TEXT_2
        track = (
            (
                f'<i class="cr-tick" style="left:{_pos(nx(floor))};width:1px;background:{WARN}"></i>'
                if floor is not None
                else ""
            )
            + f'<i class="cr-dumbbell-line" style="left:{_pos(nx(lo))};width:{nx(hi) - nx(lo):.3f}%;background:{color}"></i>'
            + f'<i class="cr-dumbbell-dot" style="left:{_pos(nx(r["mean"]))};background:{color}"></i>'
        )
        label = esc(r["label"]) + (
            '<br><span class="cr-tag cr-tag--accent">escolhido</span>' if r.get("chosen") else ""
        )
        out.append(
            _row(
                label,
                track,
                fmt(r["mean"]),
                value_sub="± " + fmt(r["se"]),
                sub=esc(r.get("sub", "")),
            )
        )
    ticks = np.linspace(d0, d1, 5)
    axis = (
        '<div class="cr-row cr-row--axis"><div></div><div>'
        + _axis([(nx(t), fmt(t)) for t in ticks])
        + "</div><div></div></div>"
    )
    return (
        legend(
            [
                ("Média na validação cruzada", ACCENT, "box"),
                ("± 1 erro-padrão", TEXT_2, "line"),
                ("Limite do empate", WARN, "line"),
            ]
        )
        + '<div class="cr-rows">'
        + "".join(out)
        + axis
        + "</div>"
    )


def logit_to_p(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
