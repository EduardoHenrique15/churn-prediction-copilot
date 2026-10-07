# 002 — Escolha do modelo pela regra de 1 erro-padrão

**Status:** aceita · **Data:** 26/09/2026

## Contexto

Quatro candidatos: baseline (taxa média), regressão logística, Random Forest e Gradient Boosting. Na validação cruzada 5 × 3 (só com dados de treino), a *average precision* média ficou em 0,660, 0,664 e 0,667 para os três modelos de verdade. Escolher o maior número trocaria de modelo a cada semente aleatória: numa versão anterior do código, a escolha virava por uma diferença de 0,00002.

## Decisão

Regra de 1 erro-padrão (Breiman et al., 1984): entre os candidatos cuja média fica a até 1 erro-padrão da melhor, vence o mais simples (regressão logística < Random Forest < Gradient Boosting).

O erro-padrão usa a correção de Nadeau & Bengio (2003) para validação cruzada repetida: os folds compartilham dados de treino, então `desvio / √n` subestima a incerteza. A correção multiplica a variância por `1/n + 1/(k−1)`.

## Alternativas consideradas

- **Maior média, sem regra:** instável, como descrito acima.
- **Teste pareado entre modelos:** mais rigoroso, mas não responde à pergunta de negócio ("vale a complexidade?") e continuaria escolhendo pelo detalhe.

## Consequências

- Vence a regressão logística: 0,007 abaixo do Gradient Boosting, dentro do erro-padrão (0,007).
- Explicação por cliente exata (sem aproximação) e API mais leve: sem SHAP, a imagem da API usa cerca de 170 MB de RAM, contra o limite de 512 MB do plano gratuito do Render.
- Se um retreino mudar a escolha para um modelo de árvore, o código continua funcionando (explicação via SHAP), e `python -m src.train` avisa para incluir o SHAP na imagem da API.
