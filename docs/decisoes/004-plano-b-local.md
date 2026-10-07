# 004 — Plano B local quando a API hiberna

**Status:** aceita · **Data:** 26/09/2026

## Contexto

A API roda no plano gratuito do Render, que desliga o serviço após 15 minutos sem tráfego; o primeiro acesso depois disso leva cerca de 1 minuto. Quem chega pelo link de um post dificilmente espera um minuto olhando um spinner.

## Decisão

A interface e o agente usam um cliente (`src/client.py`) que:

1. tenta a API com timeout curto;
2. se ela não responde, calcula a mesma previsão localmente — mesmo arquivo de modelo, mesmo código de encoding e decisão (`src/predictor.py`, que a própria API usa) — e dispara em segundo plano uma chamada que acorda a API;
3. volta a usar a API assim que ela responde.

Erros de validação nunca caem no plano B: um cliente inválido é inválido nos dois caminhos (as regras vêm de `src/schema.py`, compartilhado).

## Consequências

- A demonstração funciona na hora, mesmo com a API dormindo.
- Toda previsão diz de onde veio ("via API" ou "cálculo local"), e o chip no topo mostra o estado da API — nada escondido.
- A interface carrega o modelo também (poucos MB: é uma regressão logística).
- Alternativa descartada: um "keep-alive" pingando a API a cada 10 minutos. Funciona (750 horas grátis por mês cobrem um serviço 24 × 7), mas depende de um serviço externo e não resolve a primeira visita depois de uma queda.
