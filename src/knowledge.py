"""Documentos da base de conhecimento gerados pelo treino (04 e 05).

A versão anterior do projeto tinha esse texto escrito à mão — e ele ficou
velho: falava de outro modelo, de acurácia de 75% e de um corte de 50%
que já não existiam. Agora `python -m src.train` reescreve o arquivo com os
números do modelo que acabou de ser salvo, e o assistente sempre cita o
modelo que está no ar. (Depois de treinar, reconstrua a base vetorial com
`python -m src.build_knowledge_base` para o RAG enxergar a nova versão.)

Cada seção é curta de propósito: o indexador corta os documentos em
pedaços de ~500 caracteres, e uma seção inteira num pedaço só dá contexto
melhor ao modelo.
"""

from __future__ import annotations

from pathlib import Path

from src.business import fmt_brl, fmt_int, fmt_num, fmt_pct

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge_base"
MODEL_DOC_PATH = KNOWLEDGE_DIR / "04_sobre_o_modelo.md"
GUIDE_DOC_PATH = KNOWLEDGE_DIR / "05_como_ler_o_radar.md"

_DISPLAY = {
    "dummy_baseline": "baseline (taxa média)",
    "logistic_regression": "regressão logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
}

_WITH_ARTICLE = {
    "logistic_regression": "a regressão logística",
    "random_forest": "o Random Forest",
    "hist_gradient_boosting": "o Gradient Boosting",
}

_EXPLANATION = {
    True: (
        "Cada previsão vem com a contribuição exata de cada característica do cliente: "
        "na regressão logística, o risco em log-odds é a soma de um ponto de partida "
        "com o peso de cada variável, então a explicação não é uma aproximação."
    ),
    False: (
        "Cada previsão vem com a contribuição de cada característica do cliente, "
        "calculada com valores SHAP (TreeExplainer), que somam exatamente a saída do modelo."
    ),
}


def _ci(metrics: dict, key: str, decimals: int = 3) -> str:
    low, high = metrics["ci95"][key]
    return f"{fmt_num(low, decimals)} a {fmt_num(high, decimals)}"


def render_model_doc(metrics: dict, evaluation: dict) -> str:
    chosen = metrics["model_selected"]
    name = _DISPLAY.get(chosen, chosen)
    selection = metrics["selection"]
    best = _DISPLAY.get(selection["best_mean"], selection["best_mean"])
    cv = metrics["cv_comparison"]
    test = metrics["test_metrics"]
    op = metrics["operational"]
    threshold = evaluation["optimal_threshold"]
    cuts = evaluation["risk_level_cuts"]
    costs = evaluation["cost_assumptions"]
    redundancy = metrics.get("redundancy", {})
    excluded = metrics.get("excluded_features", [])
    linear = chosen == "logistic_regression"

    others = [_WITH_ARTICLE.get(k, k) for k in selection["tied"] if k != selection["best_mean"]]
    if selection["best_mean"] == chosen:
        selection_text = f"A {name} teve a maior average precision média e foi escolhida."
    else:
        listed = ", ".join(others[:-1]) + " e " + others[-1] if len(others) > 1 else others[0]
        selection_text = (
            f"O {best} teve a maior average precision média "
            f"({fmt_num(cv[selection['best_mean']]['cv_ap_mean'], 3)}), mas {listed} "
            "ficaram a menos de 1 erro-padrão dele. É um empate técnico, e nesse caso o "
            f"projeto fica com o modelo mais simples: {name} "
            f"(average precision de {fmt_num(cv[chosen]['cv_ap_mean'], 3)})."
        )

    excluded_text = ""
    if excluded:
        r2 = redundancy.get("monthly_r2_services")
        corr = redundancy.get("total_vs_tenure_corr")
        detail = []
        if r2 is not None:
            detail.append(
                f"a mensalidade é determinada pelos serviços contratados (R² = {fmt_num(r2, 3)})"
            )
        if corr is not None:
            detail.append(
                f"o total gasto acompanha o tempo de casa (correlação de {fmt_num(corr, 2)})"
            )
        excluded_text = (
            "\n\nMensalidade e total gasto são recebidos e validados, mas não entram no modelo: "
            + " e ".join(detail)
            + ". Com elas, a previsão não melhorava e os pesos das variáveis perdiam a leitura "
            "de negócio (a mensalidade chegava a aparecer como fator de proteção)."
        )

    sections = [
        "# Sobre o modelo do Churn Radar",
        "## Qual modelo é usado\n\n"
        f"O Churn Radar usa {name}, escolhida entre quatro candidatos: baseline (taxa média), "
        "regressão logística, Random Forest e Gradient Boosting. A comparação usou validação "
        "cruzada repetida (5 partes × 3 repetições) só com os dados de treino. " + selection_text,
        "## Quais informações o modelo usa\n\n"
        f"O modelo usa {metrics['n_features']} variáveis derivadas do perfil do cliente: tempo "
        "de casa, tipo de contrato, tipo de internet, forma de pagamento, fatura digital, "
        "serviços adicionais (segurança, backup, proteção, suporte, streaming, telefone) e "
        "perfil (idoso, cônjuge, dependentes, gênero)." + excluded_text,
        "## Desempenho no conjunto de teste\n\n"
        f"O modelo foi avaliado uma única vez em {fmt_int(metrics['n_test'])} clientes que ele "
        "nunca viu no treino:\n"
        f"- Average precision: {fmt_num(test['average_precision'], 3)} (intervalo de 95%: "
        f"{_ci(metrics, 'average_precision')}). Um modelo sem informação teria "
        f"{fmt_num(metrics['churn_rate'], 3)}, a taxa de churn da base.\n"
        f"- ROC-AUC: {fmt_num(test['roc_auc'], 3)} (intervalo de 95%: {_ci(metrics, 'roc_auc')}).\n"
        f"- Calibração: em média, a probabilidade prevista fica a {fmt_num(test['ece'] * 100, 1)} "
        "pontos percentuais da taxa real de cancelamento (ECE). Por isso a probabilidade pode "
        "ser lida ao pé da letra: entre clientes com 30% de risco, cerca de 30% cancelam.",
        "## Como o modelo decide quem contatar\n\n"
        "O modelo recomenda contatar um cliente quando o valor esperado da ação de retenção "
        "é positivo. Com as hipóteses de referência (valor do cliente de "
        f"{fmt_brl(costs['ltv'])}, oferta de {fmt_brl(costs['offer_cost'])} e "
        f"{fmt_pct(costs['success_rate'], 0)} de sucesso), isso acontece a partir de "
        f"{fmt_pct(threshold, 0)} de probabilidade — o threshold foi escolhido nos dados de "
        "treino e bate com a conta teórica (custo ÷ (sucesso × valor do cliente) = "
        f"{fmt_pct(evaluation['theoretical_threshold'], 0)}). Nesse ponto, no teste, o modelo "
        f"contata {fmt_pct(op['contact_rate'])} da base, encontra {fmt_pct(op['recall'])} dos "
        f"clientes que iam cancelar (recall) e acerta {fmt_pct(op['precision'])} das vezes "
        "(precisão). As hipóteses de custo podem ser mudadas na aba Estratégia.",
        "## Faixas de risco\n\n"
        f"- Baixo: abaixo de {fmt_pct(cuts['baixo_max'], 0)} — contatar custaria mais do que o "
        "retorno esperado.\n"
        f"- Médio: de {fmt_pct(cuts['baixo_max'], 0)} até {fmt_pct(cuts['medio_max'], 0)} — "
        "contatar já compensa.\n"
        f"- Alto: a partir de {fmt_pct(cuts['medio_max'], 0)} — o cliente tem mais chance de "
        "cancelar do que de ficar.",
        "## Como cada previsão é explicada\n\n" + _EXPLANATION[linear],
        "## Limitações\n\n"
        "- Os dados são o dataset público IBM Telco Customer Churn (7.043 clientes de uma "
        "operadora dos EUA): um retrato de um único momento, não uma operação real.\n"
        "- O modelo aprende associações, não causas. Mudar o contrato de um cliente na "
        "simulação mostra o que o modelo prevê, não o efeito garantido de uma ação.\n"
        "- Os custos da política de retenção são hipóteses de referência, não números de "
        "uma empresa.\n"
        "- Em clientes com contrato de 2 anos o churn é raro (cerca de 3%) e o modelo quase "
        "nunca recomenda contato nesse grupo.",
    ]
    return "\n\n".join(sections) + "\n"


def render_reading_guide(metrics: dict, evaluation: dict) -> str:
    """Guia de leitura da interface, com as faixas e o threshold do treino."""
    low = fmt_pct(evaluation["risk_level_cuts"]["baixo_max"], 0)
    high = fmt_pct(evaluation["risk_level_cuts"]["medio_max"], 0)
    threshold = fmt_pct(evaluation["optimal_threshold"], 0)
    costs = evaluation["cost_assumptions"]
    base_rate = fmt_pct(metrics["churn_rate"])
    sections = [
        "# Como ler o Churn Radar",
        "## Probabilidade de cancelar\n\n"
        "É a chance estimada de o cliente cancelar. Como o modelo é calibrado, o número pode "
        "ser lido ao pé da letra: entre muitos clientes com 40% de risco, cerca de 40 em cada "
        f"100 cancelam. Compare sempre com a taxa média de cancelamento da base, de {base_rate}.",
        "## Faixa de risco\n\n"
        f"- Baixo: abaixo de {low}. Contatar custaria mais do que o retorno esperado.\n"
        f"- Médio: de {low} a {high}. Contatar já compensa.\n"
        f"- Alto: a partir de {high}. O cliente tem mais chance de cancelar do que de ficar.",
        "## Recomendação de contato\n\n"
        '"Contatar" significa que o valor esperado da ação de retenção é positivo: '
        "probabilidade × taxa de sucesso da oferta × valor do cliente é maior que o custo da "
        f"oferta. Com as hipóteses de referência (valor do cliente de {fmt_brl(costs['ltv'])}, "
        f"oferta de {fmt_brl(costs['offer_cost'])} e {fmt_pct(costs['success_rate'], 0)} de "
        f"sucesso), isso acontece a partir de {threshold}.",
        "## O que mais pesou na previsão\n\n"
        "Para cada cliente, o Radar mostra quanto cada característica empurrou o risco para "
        "cima (aumenta o risco) ou para baixo (reduz o risco), partindo de um cliente de "
        "referência. Os efeitos somam exatamente a previsão final.",
        '## Simulações "e se"\n\n'
        "As simulações mostram o que o modelo prevê se uma característica mudar (por exemplo, "
        "migrar para contrato anual). Elas indicam onde vale testar uma ação, mas não garantem "
        "o resultado: o modelo aprende associações dos dados, não causas.",
        "## Análise de carteira e monitor de mudanças\n\n"
        "Na aba Carteira é possível enviar um CSV com vários clientes. O Radar valida linha "
        "por linha, ordena quem contatar primeiro e compara o perfil da carteira com os dados "
        "de treino pelo PSI (Population Stability Index). PSI abaixo de 0,10 indica perfil "
        "estável; de 0,10 a 0,25, atenção; acima de 0,25, mudança forte — nesse caso as "
        "previsões merecem mais cautela, porque a carteira é diferente dos clientes que o "
        "modelo conheceu.",
        "## Simulador de estratégia\n\n"
        "Na aba Estratégia dá para mudar o valor do cliente, o custo da oferta e a taxa de "
        "sucesso e ver o novo ponto de corte ideal e o valor líquido da campanha, calculado "
        "sobre clientes que o modelo não viu no treino.",
    ]
    return "\n\n".join(sections) + "\n"


def write_model_doc(metrics: dict, evaluation: dict, path: Path = MODEL_DOC_PATH) -> Path:
    """Reescreve os documentos que dependem do treino (modelo e guia de leitura)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_model_doc(metrics, evaluation), encoding="utf-8")
    guide = path.parent / GUIDE_DOC_PATH.name
    guide.write_text(render_reading_guide(metrics, evaluation), encoding="utf-8")
    return path
