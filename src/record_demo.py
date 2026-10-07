"""Grava as respostas de demonstração do assistente.

Rode UMA vez, com a GEMINI_API_KEY no .env e a base do RAG já construída:

    python -m src.record_demo

Usa cerca de 6 requisições da cota do Gemini (3 perguntas × 1 a 2 chamadas).
O resultado vai para data/demo/agent_demo.json, que a interface mostra
quando a cota do dia acaba — sempre com a data da gravação e o modelo usado,
para ninguém confundir uma resposta gravada com uma resposta ao vivo.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from src.agent import CHAT_MODEL, DEMO_PROMPTS, TurnResult, run_turn

DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "agent_demo.json"


def record(
    prompts: Mapping[str, Mapping[str, str]] = DEMO_PROMPTS,
    path: Path = DEMO_PATH,
    runner: Callable[[str], TurnResult] = run_turn,
) -> dict:
    """Roda cada pergunta de demonstração uma vez e salva o resultado."""
    answers = {}
    for key, item in prompts.items():
        result = runner(item["prompt"])
        answers[key] = {
            "titulo": item["titulo"],
            "pergunta": item["prompt"],
            "resposta": result.answer,
            "ferramentas": [
                {
                    "tool": step["tool"],
                    "args": step["args"],
                    "resultado": step["result"],
                    "sources": step.get("sources"),
                    "fonte": step.get("fonte"),
                    "ms": step.get("ms"),
                }
                for step in result.trace
            ],
            "estatisticas": result.stats,
        }
        print(f"  ✓ {item['titulo']} ({result.stats.get('model_calls', 0)} chamada(s) ao modelo)")

    payload = {
        "gravado_em": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "modelo": CHAT_MODEL,
        "respostas": answers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY não configurada. Defina-a no .env antes de gravar.")
    answer = input(
        f"Isto usa cerca de 6 requisições da sua cota diária do Gemini ({CHAT_MODEL}). "
        "Continuar? [s/N] "
    )
    if answer.strip().lower() not in {"s", "sim", "y", "yes"}:
        print("Nada foi gravado.")
        return
    record()
    print(f"\nRespostas salvas em {DEMO_PATH}")


if __name__ == "__main__":
    main()
