# Como ler o Churn Radar

## Probabilidade de cancelar

É a chance estimada de o cliente cancelar. Como o modelo é calibrado, o número pode ser lido ao pé da letra: entre muitos clientes com 40% de risco, cerca de 40 em cada 100 cancelam. Compare sempre com a taxa média de cancelamento da base, de 26,6%.

## Faixa de risco

- Baixo: abaixo de 32%. Contatar custaria mais do que o retorno esperado.
- Médio: de 32% a 50%. Contatar já compensa.
- Alto: a partir de 50%. O cliente tem mais chance de cancelar do que de ficar.

## Recomendação de contato

"Contatar" significa que o valor esperado da ação de retenção é positivo: probabilidade × taxa de sucesso da oferta × valor do cliente é maior que o custo da oferta. Com as hipóteses de referência (valor do cliente de R$ 1.000, oferta de R$ 100 e 30% de sucesso), isso acontece a partir de 32%.

## O que mais pesou na previsão

Para cada cliente, o Radar mostra quanto cada característica empurrou o risco para cima (aumenta o risco) ou para baixo (reduz o risco), partindo de um cliente de referência. Os efeitos somam exatamente a previsão final.

## Simulações "e se"

As simulações mostram o que o modelo prevê se uma característica mudar (por exemplo, migrar para contrato anual). Elas indicam onde vale testar uma ação, mas não garantem o resultado: o modelo aprende associações dos dados, não causas.

## Análise de carteira e monitor de mudanças

Na aba Carteira é possível enviar um CSV com vários clientes. O Radar valida linha por linha, ordena quem contatar primeiro e compara o perfil da carteira com os dados de treino pelo PSI (Population Stability Index). PSI abaixo de 0,10 indica perfil estável; de 0,10 a 0,25, atenção; acima de 0,25, mudança forte — nesse caso as previsões merecem mais cautela, porque a carteira é diferente dos clientes que o modelo conheceu.

## Simulador de estratégia

Na aba Estratégia dá para mudar o valor do cliente, o custo da oferta e a taxa de sucesso e ver o novo ponto de corte ideal e o valor líquido da campanha, calculado sobre clientes que o modelo não viu no treino.
