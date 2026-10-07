"""Página "Assistente": agente com Gemini (function calling + RAG).

A cota gratuita do Gemini é de ~20 requisições por dia e cada pergunta usa
1 ou 2. Por isso a página tem limites por sessão e por dia, e mostra
respostas GRAVADAS (com data e modelo, nunca disfarçadas de ao vivo) quando
o limite acaba ou a chave não está configurada.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import streamlit as st

from src.ui import components
from src.ui.data import (
    agent_questions_today,
    demo_answers,
    mark_quota_exhausted,
    quota_exhausted,
    register_agent_question,
)
from src.ui.html import esc, md_text, ui_html
from src.ui.settings import AGENT_MAX_QUESTIONS_PER_DAY, AGENT_MAX_QUESTIONS_PER_SESSION

logger = logging.getLogger(__name__)

TOOL_NAMES = {
    "predict_churn": "previsão do modelo",
    "search_churn_knowledge": "base de conhecimento",
}
MAX_DISPLAY_MESSAGES = 40


def _state() -> dict:
    s = st.session_state
    s.setdefault("chat", [])
    s.setdefault("agent_history", [])
    s.setdefault("agent_asked", 0)
    s.setdefault("pending_prompt", None)
    return s


def _availability() -> tuple[bool, str]:
    """(pode perguntar ao vivo?, motivo quando não pode)."""
    if not os.getenv("GEMINI_API_KEY"):
        return False, "O assistente ao vivo não está ativo nesta instância."
    if quota_exhausted():
        return False, "A cota gratuita de hoje do modelo de linguagem acabou."
    if st.session_state.agent_asked >= AGENT_MAX_QUESTIONS_PER_SESSION:
        return False, (
            f"Você já fez as {AGENT_MAX_QUESTIONS_PER_SESSION} perguntas ao vivo desta sessão."
        )
    if agent_questions_today() >= AGENT_MAX_QUESTIONS_PER_DAY:
        return False, "O limite de perguntas ao vivo de hoje foi atingido."
    return True, ""


def _date(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return "—"


def _trace_html(trace: list[dict]) -> str:
    chips = []
    for step in trace:
        name = TOOL_NAMES.get(step.get("tool"), step.get("tool", "ferramenta"))
        extra = ""
        if step.get("sources"):
            extra = " · " + ", ".join(
                s.removesuffix(".md").split("_", 1)[-1].replace("_", " ") for s in step["sources"]
            )
        elif step.get("fonte"):
            extra = " · via API" if step["fonte"] == "api" else " · cálculo local"
        ms = f" · {step['ms']} ms" if step.get("ms") is not None else ""
        chips.append(f"<span>usou <b>{esc(name)}</b>{esc(extra)}{ms}</span>")
    return f'<div class="cr-trace">{"".join(chips)}</div>' if chips else ""


def _stats_text(stats: dict) -> str:
    if not stats:
        return ""
    parts = [f"{stats.get('model_calls', 0)} chamada(s) ao modelo"]
    if stats.get("latency_ms") is not None:
        parts.append(f"{stats['latency_ms'] / 1000:.1f} s".replace(".", ","))
    tokens = stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
    if tokens:
        parts.append(f"{tokens:,} tokens".replace(",", "."))
    return " · ".join(parts)


def _render_message(msg: dict) -> None:
    with st.chat_message(
        msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    ):
        if msg["role"] == "user":
            st.markdown(md_text(msg["content"]))
            return
        if msg.get("recorded"):
            ui_html(
                f'<div class="cr-recorded">Resposta gravada em {esc(_date(msg.get("recorded_at")))} '
                f"com {esc(msg.get('model', 'o modelo'))}</div>"
            )
        if msg.get("error"):
            components.note(esc(msg["content"]), "warn")
            return
        trace = _trace_html(msg.get("trace", []))
        if trace:
            ui_html(trace)
        st.markdown(md_text(msg["content"]))
        stats = _stats_text(msg.get("stats", {}))
        if stats:
            st.caption(stats)


def _ask_live(prompt: str) -> None:
    """Pergunta ao agente de verdade, com a resposta aparecendo em tempo real."""
    from src.agent import cap_history, friendly_error, is_quota_error, run_turn

    s = _state()
    s.chat.append({"role": "user", "content": prompt})
    _render_message(s.chat[-1])
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        status = st.status("Pensando…", expanded=False)
        placeholder = st.empty()
        buffer = {"text": ""}

        def on_event(event: dict) -> None:
            if event["type"] == "text":
                buffer["text"] += event["text"]
                placeholder.markdown(md_text(buffer["text"]) + " ▌")
            elif event["type"] == "reset":
                buffer["text"] = ""
                placeholder.empty()
            elif event["type"] == "tool":
                record = event["record"]
                name = TOOL_NAMES.get(record["tool"], record["tool"])
                status.update(label=f"Consultando: {name}…")
                status.write(f"{name} · {record.get('ms', 0)} ms")

        register_agent_question()
        s.agent_asked += 1
        try:
            result = run_turn(prompt, s.agent_history, on_event=on_event)
        except Exception as exc:
            logger.warning("Falha no agente: %s: %s", type(exc).__name__, exc)
            if is_quota_error(exc):
                mark_quota_exhausted()
            status.update(label="Não foi possível responder", state="error")
            placeholder.empty()
            message = friendly_error(exc)
            components.note(esc(message), "warn")
            s.chat.append({"role": "assistant", "content": message, "error": True})
            return

        used = ", ".join(dict.fromkeys(TOOL_NAMES.get(t["tool"], t["tool"]) for t in result.trace))
        status.update(
            label=f"Ferramentas usadas: {used}" if used else "Respondido sem ferramentas",
            state="complete",
        )
        placeholder.markdown(md_text(result.answer))
        stats = _stats_text(result.stats)
        if stats:
            st.caption(stats)
    s.agent_history = cap_history(result.conversation)
    s.chat.append(
        {
            "role": "assistant",
            "content": result.answer,
            "trace": result.trace,
            "stats": result.stats,
        }
    )
    del s.chat[:-MAX_DISPLAY_MESSAGES]


def _add_recorded(key: str) -> bool:
    demo = demo_answers()
    item = demo.get("respostas", {}).get(key)
    if not item:
        return False
    s = _state()
    s.chat.append({"role": "user", "content": item["pergunta"]})
    s.chat.append(
        {
            "role": "assistant",
            "content": item["resposta"],
            "trace": [
                {
                    "tool": t["tool"],
                    "sources": t.get("sources"),
                    "fonte": t.get("fonte"),
                    "ms": t.get("ms"),
                }
                for t in item.get("ferramentas", [])
            ],
            "stats": item.get("estatisticas", {}),
            "recorded": True,
            "recorded_at": demo.get("gravado_em"),
            "model": demo.get("modelo"),
        }
    )
    return True


def _demo_cards(live: bool) -> None:
    from src.agent import DEMO_PROMPTS

    recorded = demo_answers().get("respostas", {})
    cols = st.columns(len(DEMO_PROMPTS), gap="small")
    for col, (key, item) in zip(cols, DEMO_PROMPTS.items(), strict=True):
        with col, st.container(border=True, height="stretch"):
            ui_html(
                f'<p class="cr-card-title" style="margin-bottom:6px">{esc(item["titulo"])}</p>'
                f'<p class="cr-text" style="font-size:13.5px;min-height:66px;margin:0">{esc(item["resumo"])}</p>'
            )
            label = "Perguntar" if live else "Ver resposta gravada"
            disabled = not live and key not in recorded
            if st.button(
                label, key=f"demo_{key}", width="stretch", disabled=disabled, icon=":material/send:"
            ):
                if live:
                    st.session_state.pending_prompt = item["prompt"]
                else:
                    _add_recorded(key)
                st.rerun()


def render() -> None:
    from src.agent import CHAT_MODEL

    s = _state()
    components.page_header(
        "Assistente de IA",
        "Pergunte em português",
        "Um agente com Gemini que decide sozinho quando consultar o modelo de previsão (function "
        "calling) e quando buscar na base de conhecimento do projeto (RAG). Cada resposta mostra "
        "as ferramentas que ele usou.",
    )
    live, reason = _availability()
    if live:
        left_session = AGENT_MAX_QUESTIONS_PER_SESSION - s.agent_asked
        left_day = AGENT_MAX_QUESTIONS_PER_DAY - agent_questions_today()
        mode = f"{min(left_session, left_day)} pergunta(s) ao vivo disponível(is)"
    else:
        mode = "modo demonstração · respostas gravadas"
    chips = [f"modelo {CHAT_MODEL}", "ferramentas: previsão · base de conhecimento", mode]
    ui_html(
        '<div class="cr-stack" style="margin:-6px 0 18px 0">'
        + "".join(f"<span>{esc(c)}</span>" for c in chips)
        + "</div>"
    )

    # Sem perguntas ao vivo, a primeira resposta gravada já aparece aberta:
    # quem chega vê o agente em ação sem precisar clicar.
    if not live and not s.chat and not s.get("demo_preloaded"):
        s.demo_preloaded = True
        _add_recorded("prever")
    _demo_cards(live)
    if not live:
        if demo_answers().get("respostas"):
            reason += (
                " Enquanto isso, veja as respostas gravadas acima — elas mostram o agente em "
                "ação, com as ferramentas que ele usou."
            )
        else:
            reason += " Tente de novo mais tarde."
        components.note(esc(reason))

    st.write("")
    for msg in s.chat:
        _render_message(msg)

    typed = st.chat_input(
        "Descreva um cliente ou pergunte sobre churn…"
        if live
        else "Perguntas ao vivo indisponíveis agora",
        disabled=not live,
        max_chars=1200,
    )
    prompt = typed or s.pending_prompt
    s.pending_prompt = None
    if prompt and live:
        _ask_live(prompt)
    if s.chat and st.button("Limpar conversa", icon=":material/delete_sweep:"):
        s.chat, s.agent_history = [], []
        st.rerun()
