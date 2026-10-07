# 001 — Probabilidades calibradas e corte de decisão por valor esperado

**Status:** aceita · **Data:** 26/09/2026

## Contexto

A primeira versão treinava os modelos com `class_weight="balanced"` e classificava como churn qualquer cliente acima de 50%. O reequilíbrio de classes infla as probabilidades: o modelo previa, em média, 38% de risco para uma base em que 26,6% cancelam, e no grupo de 70–80% de risco só 47% cancelavam de fato (erro de calibração, ECE, de 0,117). Com probabilidades infladas, "72% de chance de cancelar" não significa 72%, e nenhuma conta de custo feita em cima delas fecha.

## Decisão

1. Nenhum candidato usa reponderação de classes; o desbalanceamento é tratado na escolha do corte, não no treino.
2. O corte de decisão sai de uma conta de valor. Contatar um cliente custa a oferta (R$ 100); um cliente que ia cancelar aceita a oferta com 30% de chance e preserva R$ 1.000. Contatar compensa quando `p × 0,30 × 1.000 > 100`, ou seja, `p > 33%`.
3. O corte usado é o que maximiza o valor líquido nas previsões fora da amostra do treino (0,32), e a conta teórica (0,33) serve de conferência.
4. As faixas de risco derivam da mesma conta: Baixo abaixo do corte, Médio a partir dele, Alto a partir de 50%.

## Consequências

- ECE no teste caiu para 0,031: a probabilidade pode ser lida ao pé da letra, e o valor esperado de cada contato é calculado por cliente.
- O corte e as faixas mudam sozinhos se as hipóteses de custo mudarem (a página Estratégia recalcula ao vivo).
- As hipóteses (R$ 1.000, R$ 100, 30%) são ilustrativas; numa empresa real, viriam do financeiro e de um teste controlado da oferta.
