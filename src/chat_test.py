from agent import chat

print("🤖 Agente de Churn iniciado. Digite 'sair' para encerrar.\n")

while True:
    user_input = input("Você: ")
    if user_input.lower() in ["sair", "exit", "quit"]:
        break

    response = chat(user_input)
    print(f"\nAgente: {response}\n")