"""Conversa com o agente pelo terminal: `python -m src.chat_test`

Cada pergunta gasta 1 ou 2 requisições da cota do Gemini.
"""

from src.agent import cap_history, friendly_error, run_turn


def main() -> None:
    print("Assistente do Churn Radar. Digite 'sair' para encerrar.\n")
    history = []

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            result = run_turn(user_input, history)
        except Exception as exc:
            print(f"\n[erro] {friendly_error(exc)}\n({type(exc).__name__}: {exc})\n")
            continue

        for step in result.trace:
            extra = f" · fontes: {', '.join(step['sources'])}" if step.get("sources") else ""
            print(f"  ↳ {step['tool']} ({step['ms']} ms){extra}")
        stats = result.stats
        print(f"\nAssistente: {result.answer}")
        print(
            f"  [{stats['model_calls']} chamada(s) ao modelo · {stats['latency_ms']} ms · "
            f"{stats['input_tokens']}+{stats['output_tokens']} tokens]\n"
        )
        history = cap_history(result.conversation)


if __name__ == "__main__":
    main()
