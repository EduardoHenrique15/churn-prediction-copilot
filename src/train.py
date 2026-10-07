"""Treino do modelo de churn.

Rode a partir da raiz do projeto: `python -m src.train`

O que este script faz, em ordem:
1. Separa 20% dos clientes como TESTE e não toca neles até o fim.
2. Compara 4 candidatos com validação cruzada repetida (5 folds × 3) só no
   treino, medindo average precision, ROC-AUC, Brier e calibração.
3. Escolhe o modelo pela regra de 1 erro-padrão: entre os candidatos a até
   1 erro-padrão do melhor, fica o mais simples.
4. Escolhe o threshold de decisão com as previsões fora-da-amostra (OOF) do
   treino, maximizando o valor líquido de uma política de retenção.
5. Só então avalia UMA vez no teste, com intervalos de confiança por
   bootstrap, calibração, métricas por subgrupo e explicações globais.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import __version__
from src.business import (
    DEFAULT_LTV,
    DEFAULT_OFFER_COST,
    DEFAULT_SUCCESS_RATE,
    build_reference_profile,
    net_value,
    optimal_threshold,
    risk_cuts,
    theoretical_threshold,
)
from src.knowledge import write_model_doc
from src.utils import (
    CATEGORICAL_COLS,
    CATEGORY_OPTIONS,
    INTERNET_SERVICES,
    NUMERIC_COLS,
    RAW_INPUT_COLUMNS,
    TARGET_COL,
    align_columns,
    preprocess_features,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 3
N_BOOTSTRAP = 1000
SELECTION_METRIC = "average_precision"

# Da mais simples para a mais complexa — usada no desempate da regra de 1
# erro-padrão. A regressão logística é a mais simples: poucos parâmetros,
# coeficientes interpretáveis e explicação exata de cada previsão.
COMPLEXITY_ORDER = ["logistic_regression", "random_forest", "hist_gradient_boosting"]

# Variáveis que o modelo NÃO usa, apesar de a API recebê-las (e validá-las):
# - MonthlyCharges é praticamente determinada pelos serviços contratados
#   (R² = 0,999 numa regressão linear sobre as dummies de serviço). Com as
#   duas juntas, a regressão divide o efeito do preço de forma arbitrária — o
#   coeficiente da mensalidade saía NEGATIVO ("mais caro, menos churn"), sem
#   leitura de negócio. O efeito do preço continua no modelo, via serviços.
# - TotalCharges ≈ tenure × MonthlyCharges: não traz informação nova além do
#   tempo de casa e dos serviços.
# Sem as duas, a average precision na validação cruzada cai 0,002 — muito
# abaixo do erro-padrão (~0,01) — e cada coeficiente passa a ter leitura
# direta. Mesma lógica da regra de 1 erro-padrão: dentro do ruído, fica o
# mais simples. Ver docs/decisoes/003-variaveis-de-cobranca.md.
EXCLUDED_FEATURES = ("MonthlyCharges", "TotalCharges")

MODEL_DISPLAY_NAMES = {
    "dummy_baseline": "Baseline (taxa média)",
    "logistic_regression": "Regressão logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Gradient Boosting",
}


# ---------------------------------------------------------------------------
# Dados e candidatos
# ---------------------------------------------------------------------------
def load_raw_data(path: str, keep_id: bool = False) -> pd.DataFrame:
    """Carrega e limpa o dataset bruto.

    `TotalCharges` vem como texto e usa string vazia (não NaN) nos valores
    faltantes. A coerção numérica precisa vir ANTES do dropna — na ordem
    inversa o dropna não encontra nulo nenhum e as 11 linhas problemáticas
    entram no treino sem erro.
    """
    df = pd.read_csv(path)
    if not keep_id:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    print(f"Linhas descartadas por TotalCharges ausente: {n_missing}")

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def build_candidate_models() -> dict[str, object]:
    """Quatro famílias de hipótese, todas SEM class_weight="balanced".

    O reequilíbrio de classes inflava as probabilidades (o modelo anterior
    previa 38% de churn médio para uma base com 26,6%). Aqui o desbalanceamento
    é tratado onde ele importa — na escolha do threshold, pela função de
    valor — e as probabilidades ficam calibradas.
    """
    return {
        # "prior" prevê a taxa de churn do treino para todo mundo: é a régua
        # de "não aprendi nada" tanto para ranking quanto para calibração.
        "dummy_baseline": DummyClassifier(strategy="prior"),
        # Precisa de escala: a regularização penaliza coeficientes pela
        # escala bruta de cada variável (tenure 0-72 vs. dummies 0/1).
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        # Configuração regularizada (passos menores, folhas maiores): a
        # padrão sobreajusta neste dataset pequeno.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, y_proba, bins: int = 10) -> float:
    """Diferença média (ponderada) entre probabilidade prevista e taxa real."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    idx = np.clip((y_proba * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_proba[mask].mean() - y_true[mask].mean())
    return float(ece)


def reliability_bins(y_true, y_proba, bins: int = 10) -> list[dict]:
    """Dados do diagrama de confiabilidade (calibração) por faixa de 10%."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    idx = np.clip((y_proba * bins).astype(int), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append(
                {
                    "low": b / bins,
                    "high": (b + 1) / bins,
                    "n": int(mask.sum()),
                    "mean_predicted": float(y_proba[mask].mean()),
                    "observed_rate": float(y_true[mask].mean()),
                }
            )
    return out


def ranking_metrics(y_true, y_proba) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y_true, y_proba),
    }


def sweep_thresholds(y_true, y_proba) -> list[dict]:
    """Matriz de confusão completa em 101 thresholds (0,00 a 1,00).

    É essa matriz por ponto que permite recalcular o valor de QUALQUER
    política (LTV, custo e taxa de sucesso) sem rodar o modelo de novo — é o
    que o Simulador de ROI faz ao vivo.
    """
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    curve = []
    for t in np.linspace(0.0, 1.0, 101):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        curve.append(
            {
                "threshold": round(float(t), 2),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return curve


def operational_metrics(
    y_true, y_proba, threshold: float, ltv: float, offer_cost: float, success_rate: float
) -> dict[str, float]:
    """Métricas NO threshold usado em produção (não no 0,5 padrão)."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    y_pred = y_proba >= threshold
    tp = int((y_pred & (y_true == 1)).sum())
    fp = int((y_pred & (y_true == 0)).sum())
    fn = int((~y_pred & (y_true == 1)).sum())
    n = len(y_true)
    value = net_value({"tp": tp, "fp": fp}, ltv, offer_cost, success_rate)
    return {
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "contact_rate": (tp + fp) / n if n else 0.0,
        "net_value_per_1000": value / n * 1000 if n else 0.0,
    }


def bootstrap_ci(
    y_true,
    y_proba,
    threshold: float,
    ltv: float,
    offer_cost: float,
    success_rate: float,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> dict[str, list[float]]:
    """Intervalos de confiança de 95% (percentil) por reamostragem do teste.

    Mostra o quanto os números do teste poderiam variar com outra amostra de
    ~1.400 clientes — sem isso, 0,65 e 0,66 parecem diferentes quando não são.
    """
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.min() == yt.max():
            continue
        values = {
            "average_precision": average_precision_score(yt, yp),
            "roc_auc": roc_auc_score(yt, yp),
            "brier": brier_score_loss(yt, yp),
            **{
                k: v
                for k, v in operational_metrics(
                    yt, yp, threshold, ltv, offer_cost, success_rate
                ).items()
                if k != "threshold"
            },
        }
        for k, v in values.items():
            samples.setdefault(k, []).append(float(v))
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()
    }


def compute_subgroup_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float, group_masks: dict[str, np.ndarray]
) -> dict[str, dict]:
    """Métricas recortadas por subgrupo, no threshold de produção.

    A média global pode esconder um grupo em que o modelo vai bem pior. AP e
    ROC-AUC só entram quando o subgrupo tem as duas classes.
    """
    out: dict[str, dict] = {}
    for name, mask in group_masks.items():
        n = int(mask.sum())
        if n == 0:
            continue
        yt, yp = y_true[mask], y_proba[mask]
        y_pred = yp >= threshold
        tp = int((y_pred & (yt == 1)).sum())
        fp = int((y_pred & (yt == 0)).sum())
        fn = int((~y_pred & (yt == 1)).sum())
        entry = {
            "n": n,
            "churn_rate": float(yt.mean()),
            "mean_predicted": float(yp.mean()),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "contact_rate": float(y_pred.mean()),
        }
        if len(set(yt.tolist())) > 1:
            entry["roc_auc"] = float(roc_auc_score(yt, yp))
            entry["average_precision"] = float(average_precision_score(yt, yp))
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------
def corrected_standard_error(fold_scores: np.ndarray) -> float:
    """Erro-padrão da média dos folds corrigido para validação cruzada
    repetida (Nadeau & Bengio, 2003).

    Os folds NÃO são amostras independentes — os conjuntos de treino se
    sobrepõem — então std/√n subestima a incerteza. A correção multiplica a
    variância por (1/n + n_teste/n_treino), com n_teste/n_treino = 1/(k-1)
    num k-fold.
    """
    n = len(fold_scores)
    ratio = 1 / (CV_SPLITS - 1)
    return float(np.sqrt((1 / n + ratio) * np.var(fold_scores, ddof=1)))


def cross_validate_candidates(candidates: dict, X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    """AP por fold (5×3) + previsões fora-da-amostra (OOF) de cada candidato."""
    rskf = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}
    for name, model in candidates.items():
        fold_ap = []
        for train_idx, val_idx in rskf.split(X, y):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = fitted.predict_proba(X.iloc[val_idx])[:, 1]
            fold_ap.append(average_precision_score(y.iloc[val_idx], proba))
        oof = cross_val_predict(clone(model), X, y, cv=skf, method="predict_proba")[:, 1]
        fold_ap = np.asarray(fold_ap)
        results[name] = {
            "ap_mean": float(fold_ap.mean()),
            "ap_std": float(fold_ap.std(ddof=1)),
            "ap_se": corrected_standard_error(fold_ap),
            "oof": oof,
            **{f"oof_{k}": v for k, v in ranking_metrics(y, oof).items()},
        }
        print(
            f"  {name:24s} AP {results[name]['ap_mean']:.4f} ± {results[name]['ap_std']:.4f} "
            f"| Brier {results[name]['oof_brier']:.4f} | ECE {results[name]['oof_ece']:.3f}"
        )
    return results


def select_model(cv_results: dict[str, dict]) -> tuple[str, dict]:
    """Regra de 1 erro-padrão (Breiman et al., 1984): entre os candidatos
    cuja AP média fica a até 1 erro-padrão (corrigido) da melhor, escolhe o
    mais simples.

    Diferenças menores que o ruído entre folds não justificam um modelo mais
    complexo, mais pesado de servir e mais difícil de explicar.
    """
    eligible = {k: v for k, v in cv_results.items() if k in COMPLEXITY_ORDER}
    best = max(eligible, key=lambda k: eligible[k]["ap_mean"])
    floor = eligible[best]["ap_mean"] - eligible[best]["ap_se"]
    tied = [k for k in COMPLEXITY_ORDER if k in eligible and eligible[k]["ap_mean"] >= floor]
    chosen = tied[0]
    return chosen, {
        "metric": SELECTION_METRIC,
        "rule": "1 erro-padrão",
        "best_mean": best,
        "floor": floor,
        "tied": tied,
        "chosen": chosen,
    }


# ---------------------------------------------------------------------------
# Explicações globais
# ---------------------------------------------------------------------------
def linear_coefficients(model, feature_names: list[str]) -> list[dict] | None:
    """Coeficientes padronizados da regressão logística.

    Cada coeficiente é o efeito, em log-odds, de +1 desvio-padrão na variável
    (as variáveis são padronizadas no pipeline). `odds_ratio` = exp(coef):
    >1 aumenta a chance de cancelar, <1 reduz.
    """
    if not (isinstance(model, Pipeline) and hasattr(model.named_steps.get("clf"), "coef_")):
        return None
    coefs = model.named_steps["clf"].coef_[0]
    rows = [
        {"feature": f, "coef": float(c), "odds_ratio": float(np.exp(c))}
        for f, c in zip(feature_names, coefs, strict=True)
    ]
    return sorted(rows, key=lambda r: abs(r["coef"]), reverse=True)


def interpretable_effects(model, feature_columns: list[str]) -> list[dict] | None:
    """Razão de chances de cada característica, na escala original.

    Os coeficientes da regressão estão em desvios-padrão (as variáveis são
    padronizadas no pipeline) — bom para comparar pesos, ruim para ler. Aqui
    eles voltam para unidades de negócio: "+12 meses de casa" ou "fibra
    óptica em vez de DSL" (a categoria de referência de cada campo é a
    primeira em ordem alfabética, a mesma que o encoding descarta).

    Colunas que só existem juntas são somadas: "sem internet" liga também as
    seis colunas "No internet service", e "ter telefone" desliga a coluna
    "No phone service" de múltiplas linhas.
    """
    if linear_coefficients(model, feature_columns) is None:
        return None
    scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
    raw = dict(zip(feature_columns, clf.coef_[0] / scaler.scale_, strict=True))

    effects = [
        {
            "field": "tenure",
            "category": "+12 meses",
            "reference": None,
            "log_odds": raw["tenure"] * 12,
        },
        {
            "field": "SeniorCitizen",
            "category": "Yes",
            "reference": "No",
            "log_odds": raw["SeniorCitizen"],
        },
    ]
    for field, options in CATEGORY_OPTIONS.items():
        reference, *others = sorted(options)
        for category in others:
            column = f"{field}_{category}"
            if column not in raw or category in ("No internet service", "No phone service"):
                continue
            log_odds = raw[column]
            if field == "InternetService" and category == "No":
                log_odds += sum(raw.get(f"{s}_No internet service", 0.0) for s in INTERNET_SERVICES)
            if field == "PhoneService" and category == "Yes":
                log_odds -= raw.get("MultipleLines_No phone service", 0.0)
            effects.append(
                {
                    "field": field,
                    "category": category,
                    "reference": reference,
                    "log_odds": float(log_odds),
                }
            )
    for effect in effects:
        effect["log_odds"] = float(effect["log_odds"])
        effect["odds_ratio"] = float(np.exp(effect["log_odds"]))
    return sorted(effects, key=lambda e: abs(e["log_odds"]), reverse=True)


def consistent_partial_dependence(
    model,
    raw_train: pd.DataFrame,
    expected_columns: list[str],
    grids: dict[str, list[float]] | None = None,
) -> dict:
    """Risco médio previsto quando TODOS os clientes recebem o mesmo valor
    de uma variável — com TotalCharges recalculado (tenure × mensalidade)
    para não criar clientes impossíveis no caminho.
    """
    sample = raw_train.sample(n=min(1500, len(raw_train)), random_state=RANDOM_STATE)
    grids = grids or {"tenure": list(range(0, 73, 3))}
    out: dict[str, dict] = {}
    for feature, grid in grids.items():
        averages = []
        for value in grid:
            frame = sample[RAW_INPUT_COLUMNS].copy()
            frame[feature] = value
            frame["TotalCharges"] = frame["tenure"] * frame["MonthlyCharges"]
            X = align_columns(preprocess_features(frame), expected_columns)
            averages.append(float(model.predict_proba(X)[:, 1].mean()))
        out[feature] = {"grid": [float(v) for v in grid], "average": averages}
    return out


def redundancy_stats(X_full: pd.DataFrame) -> dict[str, float]:
    """Evidência para EXCLUDED_FEATURES, medida nos dados a cada treino.

    - `monthly_r2_services`: R² de uma regressão linear da mensalidade sobre
      as dummies dos serviços contratados (1,0 = totalmente determinada).
    - `total_vs_tenure_corr`: correlação entre total gasto e tempo de casa.
    """
    from sklearn.linear_model import LinearRegression

    service_prefixes = ("PhoneService", "MultipleLines", "InternetService", *INTERNET_SERVICES)
    services = [c for c in X_full.columns if c.startswith(service_prefixes)]
    monthly = X_full["MonthlyCharges"]
    r2 = LinearRegression().fit(X_full[services], monthly).score(X_full[services], monthly)
    corr = float(np.corrcoef(X_full["tenure"], X_full["TotalCharges"])[0, 1])
    return {"monthly_r2_services": float(r2), "total_vs_tenure_corr": corr}


def git_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    # MLflow só é importado aqui: os testes das funções deste módulo (e o CI)
    # não precisam dele instalado nem pagam o tempo de import.
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("churn-prediction")

    raw = load_raw_data(RAW_DATA_PATH)
    X_full = preprocess_features(raw[RAW_INPUT_COLUMNS])
    redundancy = redundancy_stats(X_full)
    X_all = X_full.drop(columns=list(EXCLUDED_FEATURES))
    print(
        f"Mensalidade explicada pelos serviços: R² = {redundancy['monthly_r2_services']:.3f} "
        f"| correlação total gasto × tempo de casa: {redundancy['total_vs_tenure_corr']:.2f}"
    )
    y_all = raw[TARGET_COL].astype(int)
    feature_columns = list(X_all.columns)

    idx_train, idx_test = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )
    X_train, X_test = X_all.loc[idx_train], X_all.loc[idx_test]
    y_train, y_test = y_all.loc[idx_train], y_all.loc[idx_test]
    raw_train, raw_test = raw.loc[idx_train], raw.loc[idx_test]
    print(f"Treino: {len(X_train)} clientes | Teste: {len(X_test)} clientes")

    print("\nValidação cruzada (5 folds × 3 repetições, só no treino):")
    candidates = build_candidate_models()
    cv = cross_validate_candidates(candidates, X_train, y_train)
    chosen, selection = select_model(cv)
    print(
        f"\nMelhor AP média: {selection['best_mean']} — empatados dentro de 1 erro-padrão: "
        f"{selection['tied']} → escolhido: {chosen}"
    )

    for name, result in cv.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", type(candidates[name]).__name__)
            mlflow.log_param("selected", name == chosen)
            mlflow.log_metrics(
                {
                    "cv_ap_mean": result["ap_mean"],
                    "cv_ap_std": result["ap_std"],
                    "oof_roc_auc": result["oof_roc_auc"],
                    "oof_brier": result["oof_brier"],
                    "oof_ece": result["oof_ece"],
                }
            )

    model = clone(candidates[chosen]).fit(X_train, y_train)

    # Threshold escolhido nas previsões OOF do TREINO — o teste continua
    # intocado até a avaliação final.
    oof = cv[chosen]["oof"]
    curve_oof = sweep_thresholds(y_train, oof)
    threshold = optimal_threshold(curve_oof, DEFAULT_LTV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE)
    cuts = risk_cuts(threshold)
    theory = theoretical_threshold(DEFAULT_LTV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE)
    print(f"Threshold escolhido (OOF): {threshold:.2f} | teórico: {theory:.2f}")

    # Avaliação final — a única vez em que o teste é usado.
    p_test = model.predict_proba(X_test)[:, 1]
    yt = y_test.to_numpy()
    test_metrics = ranking_metrics(yt, p_test)
    operational = operational_metrics(
        yt, p_test, threshold, DEFAULT_LTV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE
    )
    ci = bootstrap_ci(yt, p_test, threshold, DEFAULT_LTV, DEFAULT_OFFER_COST, DEFAULT_SUCCESS_RATE)
    curve_test = sweep_thresholds(yt, p_test)
    print(
        f"Teste: AP {test_metrics['average_precision']:.3f} | ROC-AUC {test_metrics['roc_auc']:.3f} "
        f"| Brier {test_metrics['brier']:.4f} | ECE {test_metrics['ece']:.3f}"
    )
    print(
        f"No threshold {threshold:.2f}: precision {operational['precision']:.1%} | "
        f"recall {operational['recall']:.1%} | contata {operational['contact_rate']:.1%} da base"
    )

    senior = raw_test["SeniorCitizen"].to_numpy() == 1
    contract = raw_test["Contract"].to_numpy()
    subgroups = compute_subgroup_metrics(
        yt,
        p_test,
        threshold,
        {
            "senior_citizen": senior,
            "non_senior_citizen": ~senior,
            "contract_month_to_month": contract == "Month-to-month",
            "contract_one_year": contract == "One year",
            "contract_two_year": contract == "Two year",
        },
    )

    perm = permutation_importance(
        model, X_test, y_test, scoring=SELECTION_METRIC, n_repeats=10, random_state=RANDOM_STATE
    )
    top_features = [
        {"feature": f, "importance": float(v)}
        for f, v in sorted(
            zip(feature_columns, perm.importances_mean, strict=True), key=lambda t: -t[1]
        )[:8]
        if v > 0
    ]

    trained_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    metadata = {
        "model_version": __version__,
        "trained_at": trained_at,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    pd.DataFrame({"y_true": yt, "y_proba": p_test}).to_csv(
        os.path.join(MODELS_DIR, "test_predictions.csv"), index=False
    )

    cv_table = {
        name: {
            "cv_ap_mean": r["ap_mean"],
            "cv_ap_std": r["ap_std"],
            "cv_ap_se": r["ap_se"],
            "oof_roc_auc": r["oof_roc_auc"],
            "oof_brier": r["oof_brier"],
            "oof_ece": r["oof_ece"],
        }
        for name, r in cv.items()
    }

    metrics = {
        "schema_version": 2,
        **metadata,
        "model_selected": chosen,
        "model_display_name": MODEL_DISPLAY_NAMES[chosen],
        "selection": selection,
        "n_samples": int(len(raw)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_columns),
        "excluded_features": list(EXCLUDED_FEATURES),
        "redundancy": redundancy,
        "churn_rate": float(y_all.mean()),
        "test_metrics": test_metrics,
        "operational": operational,
        "ci95": ci,
        "cv_comparison": cv_table,
        "top_features": top_features,
    }
    evaluation = {
        "schema_version": 2,
        **metadata,
        "model_selected": chosen,
        "cost_assumptions": {
            "ltv": DEFAULT_LTV,
            "offer_cost": DEFAULT_OFFER_COST,
            "success_rate": DEFAULT_SUCCESS_RATE,
            "note": "Hipóteses de referência, não medidas de uma empresa real — ajustáveis no Simulador de ROI.",
        },
        "optimal_threshold": threshold,
        "theoretical_threshold": theory,
        "risk_level_cuts": cuts,
        "curves": {"oof": curve_oof, "test": curve_test},
        "reliability": {"test": reliability_bins(yt, p_test), "ece": test_metrics["ece"]},
        "subgroup_metrics": subgroups,
        "coefficients": linear_coefficients(model, feature_columns),
        "effects": interpretable_effects(model, feature_columns),
        "partial_dependence": consistent_partial_dependence(model, raw_train, feature_columns),
    }
    profile = build_reference_profile(raw_train, CATEGORICAL_COLS + ["SeniorCitizen"], NUMERIC_COLS)

    for filename, payload in (
        ("metrics.json", metrics),
        ("evaluation.json", evaluation),
        ("reference_profile.json", profile),
    ):
        with open(os.path.join(MODELS_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    doc_path = write_model_doc(metrics, evaluation)
    print(f"Base de conhecimento atualizada: {os.path.relpath(doc_path, BASE_DIR)}")

    with mlflow.start_run(run_name=f"final_{chosen}"):
        mlflow.log_params({"model": chosen, "threshold": threshold, **cuts})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"op_{k}": v for k, v in operational.items()})
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(
                X_train.astype(float), model.predict_proba(X_train.head(5))[:, 1]
            ),
            input_example=X_train.head(1).astype(float),
            # skops (formato seguro do MLflow) exige declarar os tipos das
            # árvores do scikit-learn; o arquivo é gerado aqui mesmo.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    print(f"\nModelo salvo em models/churn_model.pkl ({chosen}, versão {__version__})")
    print(
        "Próximos passos: rode `python -m src.build_knowledge_base` para o assistente "
        "enxergar os números novos."
    )
    if linear_coefficients(model, feature_columns) is None:
        print(
            "ATENÇÃO: o modelo escolhido é de árvore — a explicação por cliente precisa do "
            "SHAP. Confira se `shap` está em requirements-api.txt antes do deploy."
        )


if __name__ == "__main__":
    main()
