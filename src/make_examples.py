"""Gera os CSVs de exemplo da aba Carteira a partir do conjunto de TESTE.

Rode a partir da raiz do projeto: `python -m src.make_examples`

Os dois arquivos saem só de clientes que o modelo nunca viu no treino (a
mesma divisão treino/teste de src/train.py), então as previsões mostradas
na demonstração são fora da amostra — e a coluna `Churn` real permite
conferir o modelo na própria interface.

- data/exemplo_lote.csv: 200 clientes sorteados do teste (perfil típico).
- data/exemplo_lote_novos.csv: 200 clientes com até 12 meses de casa. É
  uma base propositalmente diferente da de treino, para demonstrar o
  monitor de drift (PSI).
"""

from __future__ import annotations

import os

from sklearn.model_selection import train_test_split

from src.train import RANDOM_STATE, RAW_DATA_PATH, TEST_SIZE, load_raw_data
from src.utils import RAW_INPUT_COLUMNS, TARGET_COL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
N_ROWS = 200


def main() -> None:
    raw = load_raw_data(RAW_DATA_PATH, keep_id=True)
    _, idx_test = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=raw[TARGET_COL]
    )
    test = raw.loc[idx_test].copy()
    test[TARGET_COL] = test[TARGET_COL].map({1: "Yes", 0: "No"})
    columns = ["customerID", *RAW_INPUT_COLUMNS, TARGET_COL]

    typical = test.sample(n=N_ROWS, random_state=RANDOM_STATE)[columns]
    newcomers = test[test["tenure"] <= 12].sample(n=N_ROWS, random_state=RANDOM_STATE)[columns]

    for name, frame in (("exemplo_lote.csv", typical), ("exemplo_lote_novos.csv", newcomers)):
        path = os.path.join(DATA_DIR, name)
        frame.to_csv(path, index=False)
        churn = (frame[TARGET_COL] == "Yes").mean()
        print(f"{name}: {len(frame)} clientes, churn real de {churn:.1%} → {path}")


if __name__ == "__main__":
    main()