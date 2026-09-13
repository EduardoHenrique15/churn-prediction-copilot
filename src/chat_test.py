"""Teste do agente via terminal: `python -m src.chat_test`"""

from src.agent import chat


def main() -> None:
    print("Agente de Churn iniciado. Digite 'sair' para encerrar.\n")
    history = []

    while True:
        user_input = input("Você: ")
        if user_input.lower() in {"sair", "exit", "quit"}:
            break

        response, history = chat(user_input, history)
        print(f"\nAgente: {response}\n")


if __name__ == "__main__":
    main()