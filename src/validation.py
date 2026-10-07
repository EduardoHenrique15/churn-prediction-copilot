"""Validação linha a linha de CSVs em lote, com mensagens em português.

A API valida o lote inteiro de uma vez e recusa tudo se UMA linha estiver
errada. Na interface, quem sobe um arquivo precisa saber QUAIS linhas têm
problema e por quê — e ver a previsão das demais. As regras são as mesmas
da API (domínio em src/utils.py, consistência em consistency_errors); um
teste garante que as duas concordam.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from src.labels import FIELD_LABELS
from src.utils import (
    CATEGORICAL_COLS,
    CATEGORY_OPTIONS,
    NUMERIC_LIMITS,
    RAW_INPUT_COLUMNS,
    as_number,
    consistency_errors,
)

# Colunas que a própria interface acrescenta ao exportar o resultado. Se
# alguém reenviar um arquivo já pontuado, elas são descartadas antes de
# prever de novo (senão o resultado sairia com colunas duplicadas).
RESULT_COLUMNS = (
    "probabilidade_churn",
    "risco",
    "contatar",
    "valor_esperado_contato",
    "prioridade",
    "fonte_previsao",
)

_YES_NO_NUMBERS = {"yes": 1, "no": 0, "sim": 1, "não": 0, "nao": 0}


@dataclass
class RowCheck:
    record: dict | None
    errors: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


def _blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _label(col: str) -> str:
    return FIELD_LABELS.get(col, col)


def _accepted(col: str) -> str:
    options = list(CATEGORY_OPTIONS[col])
    return ", ".join(options[:-1]) + " ou " + options[-1]


def check_row(raw: Mapping) -> RowCheck:
    """Valida e normaliza um cliente cru (uma linha do CSV)."""
    errors: list[str] = []
    fixes: list[str] = []
    record: dict = {}

    for col in CATEGORICAL_COLS:
        value = raw.get(col)
        if _blank(value):
            errors.append(f"{_label(col)}: vazio")
            continue
        text = str(value).strip()
        if text not in CATEGORY_OPTIONS[col]:
            errors.append(f"{_label(col)}: '{text}' não é aceito (use {_accepted(col)})")
            continue
        record[col] = text

    senior_raw = raw.get("SeniorCitizen")
    senior = as_number(senior_raw)
    if senior is None and isinstance(senior_raw, str):
        senior = _YES_NO_NUMBERS.get(senior_raw.strip().lower())
    if senior not in (0, 1):
        errors.append(f"{_label('SeniorCitizen')}: use 0 (não) ou 1 (sim)")
    else:
        record["SeniorCitizen"] = int(senior)

    tenure = as_number(raw.get("tenure"))
    low, high = NUMERIC_LIMITS["tenure"]
    if tenure is None or tenure != int(tenure) or not low <= tenure <= high:
        errors.append(f"{_label('tenure')}: precisa ser um número inteiro de meses entre 0 e 120")
    else:
        record["tenure"] = int(tenure)

    for col in ("MonthlyCharges", "TotalCharges"):
        value = as_number(raw.get(col))
        low, high = NUMERIC_LIMITS[col]
        if value is None and col == "TotalCharges" and record.get("tenure") == 0:
            # No dataset original, clientes com 0 mês de casa têm o total
            # gasto em branco: ainda não pagaram nada.
            record[col] = 0.0
            fixes.append("Total gasto vazio com 0 mês de casa: considerado R$ 0")
            continue
        if value is None:
            errors.append(f"{_label(col)}: vazio ou não numérico")
        elif not low <= value <= high:
            errors.append(f"{_label(col)}: fora da faixa aceita ({low:g} a {high:g})")
        else:
            record[col] = float(value)

    if not errors:
        errors.extend(consistency_errors(record))
    ordered = {col: record[col] for col in RAW_INPUT_COLUMNS} if not errors else None
    return RowCheck(ordered, errors, fixes)


def prepare_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame, list[str]]:
    """Separa um CSV em linhas prontas para prever e linhas com problema.

    Devolve (dados originais limpos, registros válidos, relatório de
    problemas, avisos gerais). O relatório tem uma linha por linha do
    arquivo com problema — numerada como no editor de planilhas (cabeçalho
    é a linha 1).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    notes: list[str] = []
    dropped = [c for c in df.columns if c in RESULT_COLUMNS]
    if dropped:
        df = df.drop(columns=dropped)
        notes.append("Colunas de um resultado anterior foram ignoradas: " + ", ".join(dropped))

    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Faltam colunas obrigatórias: " + ", ".join(missing))

    valid_records: list[dict] = []
    valid_index: list = []
    problems: list[dict] = []
    fixes_count = 0
    for position, (index, row) in enumerate(df.iterrows()):
        check = check_row(row.to_dict())
        if check.record is None:
            problems.append({"linha": position + 2, "problemas": "; ".join(check.errors)})
            continue
        fixes_count += bool(check.fixes)
        valid_records.append(check.record)
        valid_index.append(index)

    if fixes_count:
        notes.append(
            f"{fixes_count} cliente(s) com 0 mês de casa e total gasto vazio: considerado R$ 0."
        )
    return df.loc[valid_index], valid_records, pd.DataFrame(problems), notes