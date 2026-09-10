# Sobre o Modelo de Previsão

O modelo utilizado é um RandomForestClassifier, treinado com dados históricos de
clientes de telecomunicações. Ele analisa 30 variáveis relacionadas ao perfil
demográfico, serviços contratados e dados de conta do cliente.

## Desempenho do modelo
- Acurácia: 75%
- Recall (detecção de clientes que realmente cancelam): 77%
- Precision: 52%

O modelo foi ajustado para priorizar recall em vez de precision, já que o custo
de não identificar um cliente em risco de cancelamento é considerado maior do
que o custo de contatar preventivamente um cliente que não cancelaria.

## Classificação de risco
- Probabilidade abaixo de 50%: risco classificado como "Baixo"
- Probabilidade igual ou acima de 50%: risco classificado como "Alto"