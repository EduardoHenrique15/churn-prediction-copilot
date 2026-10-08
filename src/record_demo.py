"""Grava as respostas de demonstração do assistente.

Rode com a GEMINI_API_KEY no .env (a base do RAG é montada sozinha na
primeira busca, se ainda não existir):

    python -m src.record_demo            # grava só as respostas que faltam
    python -m src.record_demo --refazer  # regrava todas

Usa cerca de 6 requisições da cota do Gemini (3 perguntas × 1 a 2 chamadas).
O resultado vai para data/demo/agent_demo.json, que a interface mostra
quando a cota do dia acaba — sempre com a data da gravação e o modelo usado,
para ninguém confundir uma resposta gravada com uma resposta ao vivo.

O arquivo é salvo depois de CADA resposta: se a cota acabar no meio, o que
já foi gravado (e pago em cota) não se perde, e a próxima execução grava só
o que faltou.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from src.agent import CHAT_MODEL, DEMO_PROMPTS, TurnResult, run_turn

DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "demo" / "agent_demo.json"


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save(path: Path, answers: dict) -> dict:
    payload = {
        "gravado_em": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "modelo": CHAT_MODEL,
        "respostas": answers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def record(
    prompts: Mapping[str, Mapping[str, str]] = DEMO_PROMPTS,
    path: Path = DEMO_PATH,
    runner: Callable[[str], TurnResult] = run_turn,
    redo: bool = False,
) -> dict:
    """Roda cada pergunta de demonstração e salva o arquivo a cada resposta.

    Reaproveita as respostas já gravadas com o MESMO modelo (a interface
    mostra um modelo só para todas); `redo=True` regrava tudo.
    """
    previous = _load(path)
    answers: dict = {}
    if not redo and previous.get("modelo") == CHAT_MODEL:
        answers = {k: v for k, v in previous.get("respostas", {}).items() if k in prompts}
    payload = previous if answers else {}

    for key, item in prompts.items():
        if key in answers:
            print(f"  = {item['titulo']} (já gravada)")
            continue
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
        payload = _save(path, answers)
        print(f"  ✓ {item['titulo']} ({result.stats.get('model_calls', 0)} chamada(s) ao modelo)")
    return payload


def main() -> None:
    from src.agent import friendly_error

    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY não configurada. Defina-a no .env antes de gravar.")
    redo = "--refazer" in sys.argv[1:]
    previous = _load(DEMO_PATH)
    done = set()
    if not redo and previous.get("modelo") == CHAT_MODEL:
        done = set(previous.get("respostas", {})) & set(DEMO_PROMPTS)
    missing = len(DEMO_PROMPTS) - len(done)
    if not missing:
        print(f"As {len(DEMO_PROMPTS)} respostas já estão gravadas. Use --refazer para regravar.")
        return
    answer = input(
        f"Vou gravar {missing} resposta(s): cerca de {2 * missing} requisições da sua cota "
        f"diária do Gemini ({CHAT_MODEL}). Continuar? [s/N] "
    )
    if answer.strip().lower() not in {"s", "sim", "y", "yes"}:
        print("Nada foi gravado.")
        return
    try:
        record(redo=redo)
    except Exception as exc:
        saved = len(_load(DEMO_PATH).get("respostas", {}))
        raise SystemExit(
            f"\n{friendly_error(exc)}\n({type(exc).__name__}: {exc})\n"
            f"{saved} de {len(DEMO_PROMPTS)} respostas estão salvas em {DEMO_PATH}. "
            "Rode de novo mais tarde para gravar as que faltam."
        ) from exc
    print(f"\nRespostas salvas em {DEMO_PATH}")


if __name__ == "__main__":
    main()
