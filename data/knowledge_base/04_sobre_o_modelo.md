# Sobre o modelo do Churn Radar

## Qual modelo é usado

O Churn Radar usa regressão logística, escolhida entre quatro candidatos: baseline (taxa média), regressão logística, Random Forest e Gradient Boosting. A comparação usou validação cruzada repetida (5 partes × 3 repetições) só com os dados de treino. O Gradient Boosting teve a maior average precision média (0,667), mas a regressão logística e o Random Forest ficaram a menos de 1 erro-padrão dele. É um empate técnico, e nesse caso o projeto fica com o modelo mais simples: regressão logística (average precision de 0,660).

## Quais informações o modelo usa

O modelo usa 28 variáveis derivadas do perfil do cliente: tempo de casa, tipo de contrato, tipo de internet, forma de pagamento, fatura digital, serviços adicionais (segurança, backup, proteção, suporte, streaming, telefone) e perfil (idoso, cônjuge, dependentes, gênero).

Mensalidade e total gasto são recebidos e validados, mas não entram no modelo: a mensalidade é determinada pelos serviços contratados (R² = 0,999) e o total gasto acompanha o tempo de casa (correlação de 0,83). Com elas, a previsão não melhorava e os pesos das variáveis perdiam a leitura de negócio (a mensalidade chegava a aparecer como fator de proteção).

## Desempenho no conjunto de teste

O modelo foi avaliado uma única vez em 1.407 clientes que ele nunca viu no treino:
- Average precision: 0,619 (intervalo de 95%: 0,566 a 0,673). Um modelo sem informação teria 0,266, a taxa de churn da base.
- ROC-AUC: 0,834 (intervalo de 95%: 0,812 a 0,855).
- Calibração: em média, a probabilidade prevista fica a 3,1 pontos percentuais da taxa real de cancelamento (ECE). Por isso a probabilidade pode ser lida ao pé da letra: entre clientes com 30% de risco, cerca de 30% cancelam.

## Como o modelo decide quem contatar

O modelo recomenda contatar um cliente quando o valor esperado da ação de retenção é positivo. Com as hipóteses de referência (valor do cliente de R$ 1.000, oferta de R$ 100 e 30% de sucesso), isso acontece a partir de 32% de probabilidade — o threshold foi escolhido nos dados de treino e bate com a conta teórica (custo ÷ (sucesso × valor do cliente) = 33%). Nesse ponto, no teste, o modelo contata 36,5% da base, encontra 73,0% dos clientes que iam cancelar (recall) e acerta 53,2% das vezes (precisão). As hipóteses de custo podem ser mudadas na aba Estratégia.

## Faixas de risco

- Baixo: abaixo de 32% — contatar custaria mais do que o retorno esperado.
- Médio: de 32% até 50% — contatar já compensa.
- Alto: a partir de 50% — o cliente tem mais chance de cancelar do que de ficar.

## Como cada previsão é explicada

Cada previsão vem com a contribuição exata de cada característica do cliente: na regressão logística, o risco em log-odds é a soma de um ponto de partida com o peso de cada variável, então a explicação não é uma aproximação.

## Limitações

- Os dados são o dataset público IBM Telco Customer Churn (7.032 clientes de uma operadora dos EUA): um retrato de um único momento, não uma operação real.
- O modelo aprende associações, não causas. Mudar o contrato de um cliente na simulação mostra o que o modelo prevê, não o efeito garantido de uma ação.
- Os custos da política de retenção são hipóteses de referência, não números de uma empresa.
- Em clientes com contrato de 2 anos o churn é raro (2,4% no conjunto de teste) e o modelo quase nunca recomenda contato nesse grupo.
