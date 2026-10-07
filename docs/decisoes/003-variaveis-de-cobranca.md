# 003 — Mensalidade e total gasto fora do modelo

**Status:** aceita · **Data:** 26/09/2026

## Contexto

Com a regressão logística escolhida, os coeficientes padronizados mostravam a mensalidade com peso **negativo** (−0,85): "quanto mais caro, menos churn". A explicação de um cliente de fibra com mensalidade alta dizia que a mensalidade *reduzia* o risco — o oposto do que os dados mostram (quem cancela paga R$ 74 por mês em média; quem fica, R$ 61).

O motivo é colinearidade: a mensalidade é praticamente determinada pelos serviços contratados. Uma regressão linear da mensalidade sobre as colunas de serviço tem **R² = 0,999**. Com mensalidade e serviços no mesmo modelo, a regressão divide o efeito do preço entre as colunas de forma arbitrária. O total gasto, por sua vez, é quase tempo de casa × mensalidade (correlação de 0,83 com o tempo de casa).

## Decisão

As duas variáveis saem do modelo (`EXCLUDED_FEATURES` em `src/train.py`). A API continua recebendo e validando os dois campos — o contrato não muda, e eles seguem úteis para a regra de consistência do total gasto e para a receita em risco mostrada na interface.

## Evidência

Validação cruzada 5 × 3 da regressão logística, só com dados de treino:

| Variáveis | Average precision | ROC-AUC |
|---|---|---|
| Todas | 0,6622 | 0,8472 |
| Sem mensalidade | 0,6625 | 0,8470 |
| Sem mensalidade e sem total gasto | 0,6600 | 0,8454 |

A diferença (0,002) está muito abaixo do erro-padrão (~0,01). Mesma lógica da decisão 002: dentro do ruído, fica o mais simples. O R² e a correlação são recalculados a cada treino e salvos em `models/metrics.json`.

## Consequências

- Cada peso do modelo tem leitura de negócio: fibra óptica ×2,5 na chance de cancelar contra DSL; contrato de dois anos ×0,25 contra mensal.
- O efeito do preço continua no modelo, pelos serviços que o compõem.
- Mudar a mensalidade no formulário não muda a previsão, e a interface diz isso ao lado do campo.
