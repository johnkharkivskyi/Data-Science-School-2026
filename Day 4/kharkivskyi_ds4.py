"""
Приклад-шаблон розв'язку.

Скопіюйте цей файл, назвіть <прізвище>_ds<N>.py і реалізуйте у ньому дві функції:
    train_model(train_df) -> state
    predict(state, eval_df) -> np.ndarray   (довжина = len(eval_df), той самий порядок рядків)

Що вам дано (у теці out/for_students/):
    ds_<N>_train.csv  — тренувальні дані з таргетом
    ds_<N>_val.csv    — валідаційні дані з таргетом (300 рядків)

Ваше завдання — навчити модель і зробити так, щоб вона добре працювала на
ПРИХОВАНИХ даних, яких ви не бачите. Оцінювання проходить так: інструктор
викличе ваші train_model()/predict() на прихованому тестовому наборі й порахує
метрику (R² для регресії, ROC-AUC для класифікації).

Правила:
    - дозволені лише numpy, pandas, scipy, scikit-learn;
    - зафіксуйте всі сіди (див. блок нижче) — два запуски мають давати однаковий результат;
    - для класифікації predict() повертає ЙМОВІРНІСТЬ позитивного класу;
    - не читайте файли всередині train_model()/predict() — усе приходить через аргументи.

Запустіть цей файл (python <ім'я>.py), щоб перевірити, що ваш розв'язок
відповідає контракту й рахується метрика на валідації.
"""

import os
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Фіксація всіх джерел випадковості (обов'язково для відтворюваності)
# ---------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Налаштування варіанту (заповнюється під конкретний ds_N)
# ---------------------------------------------------------------------------
DS_ID = 4                        # номер вашого варіанту -> ds_1, ds_2, ...
TARGET = "SeriousDlqin2yrs"      # цільовий стовпчик вашого варіанту
TASK = "classification"          # "regression" | "classification"

DATA_DIR = os.path.join("out", "for_students")


def engineer_features(df: pd.DataFrame, income_median: float = None):
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    p30 = df["NumberOfTime30-59DaysPastDueNotWorse"]
    p60 = df["NumberOfTime60-89DaysPastDueNotWorse"]
    p90 = df["NumberOfTimes90DaysLate"]

    df["has_96_or_98"] = ((p30 >= 96) | (p60 >= 96) | (p90 >= 96)).astype(float)

    p30_c = p30.replace({96: 0, 98: 0})
    p60_c = p60.replace({96: 0, 98: 0})
    p90_c = p90.replace({96: 0, 98: 0})

    df["TotalLateCount"] = p30_c + p60_c + p90_c
    df["WeightedLateCount"] = p30_c * 1.0 + p60_c * 2.0 + p90_c * 3.0
    df["HasEverBeenLate"] = (df["TotalLateCount"] > 0).astype(float)

    if income_median is None:
        income_median = float(df["MonthlyIncome"].median())

    df["MonthlyIncome_is_na"] = df["MonthlyIncome"].isna().astype(float)
    df["NumberOfDependents_is_na"] = df["NumberOfDependents"].isna().astype(float)

    inc_filled = df["MonthlyIncome"].fillna(income_median)
    deps_filled = df["NumberOfDependents"].fillna(0)

    df["IncomePerPerson"] = inc_filled / (deps_filled + 1.0)
    df["EstimatedDebt"] = df["DebtRatio"] * inc_filled

    rev = df["RevolvingUtilizationOfUnsecuredLines"]
    df["Revolving_gt_1"] = (rev > 1.0).astype(float)
    df["Revolving_gt_10"] = (rev > 10.0).astype(float)
    df["Revolving_is_zero"] = (rev == 0).astype(float)

    open_lines = df["NumberOfOpenCreditLinesAndLoans"]
    real_estate = df["NumberRealEstateLoansOrLines"]
    df["TotalLines"] = open_lines + real_estate
    df["RealEstateShare"] = real_estate / (df["TotalLines"] + 1e-5)

    df["Log_Revolving"] = np.log1p(np.clip(rev, 0, None))
    df["Log_DebtRatio"] = np.log1p(np.clip(df["DebtRatio"], 0, None))
    df["Log_MonthlyIncome"] = np.log1p(np.clip(inc_filled, 0, None))
    df["Log_EstimatedDebt"] = np.log1p(np.clip(df["EstimatedDebt"], 0, None))

    return df, income_median


# ---------------------------------------------------------------------------
# Контракт рішення
# ---------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame):
    """Отримує train (з таргетом). Робить весь препроцесинг + навчання.
    Повертає state (модель / словник / кортеж) для передачі в predict()."""
    y_train = train_df[TARGET].values
    X_raw = train_df.drop(columns=[TARGET])
    X_fe, income_median = engineer_features(X_raw)

    feature_names = list(X_fe.columns)
    medians = X_fe.median(numeric_only=True)

    hgb1 = HistGradientBoostingClassifier(
        random_state=SEED,
        max_iter=400,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
    )

    hgb2 = HistGradientBoostingClassifier(
        random_state=SEED + 100,
        max_iter=500,
        learning_rate=0.02,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    hgb1.fit(X_fe, y_train)
    hgb2.fit(X_fe, y_train)
    et.fit(X_fe.fillna(medians), y_train)

    return {
        "hgb1": hgb1,
        "hgb2": hgb2,
        "et": et,
        "weights": (0.35, 0.45, 0.20),
        "feature_names": feature_names,
        "medians": medians,
        "income_median": income_median,
    }


def predict(state, eval_df: pd.DataFrame) -> np.ndarray:
    """Отримує state і eval-набір БЕЗ таргету. Повертає 1-D np.ndarray.
    Для класифікації — ймовірність позитивного класу (для ROC-AUC)."""
    X_eval_fe, _ = engineer_features(eval_df, income_median=state["income_median"])
    feature_names = state["feature_names"]
    X_eval_fe = X_eval_fe[feature_names]

    medians = state["medians"]
    hgb1 = state["hgb1"]
    hgb2 = state["hgb2"]
    et = state["et"]
    w1, w2, w3 = state["weights"]

    p1 = hgb1.predict_proba(X_eval_fe)[:, 1]
    p2 = hgb2.predict_proba(X_eval_fe)[:, 1]
    p3 = et.predict_proba(X_eval_fe.fillna(medians))[:, 1]

    return np.asarray(w1 * p1 + w2 * p2 + w3 * p3)


# ---------------------------------------------------------------------------
# Локальна перевірка: чи відповідає розв'язок контракту + метрика на валідації
# ---------------------------------------------------------------------------
def _score(y_true, y_pred):
    if TASK == "regression":
        return r2_score(y_true, y_pred)   # R²
    return roc_auc_score(y_true, y_pred)


def evaluate_stratified_cv(train_df: pd.DataFrame, n_splits: int = 5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y = train_df[TARGET].values
    oof_preds = np.zeros(len(train_df), dtype=float)

    for tr_idx, va_idx in skf.split(train_df, y):
        tr_sub = train_df.iloc[tr_idx].copy()
        va_sub = train_df.iloc[va_idx].drop(columns=[TARGET]).copy()

        st = train_model(tr_sub)
        p_val = predict(st, va_sub)
        oof_preds[va_idx] = p_val

    return float(roc_auc_score(y, oof_preds))


if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, f"ds_{DS_ID}_train.csv")
    val_path = os.path.join(DATA_DIR, f"ds_{DS_ID}_val.csv")

    if not os.path.exists(train_path):
        train_path = f"ds_{DS_ID}_train.csv"
        val_path = f"ds_{DS_ID}_val.csv"
    if not os.path.exists(train_path):
        train_path = os.path.join("Day 4", f"ds_{DS_ID}_train.csv")
        val_path = os.path.join("Day 4", f"ds_{DS_ID}_val.csv")

    print(f"Loading data from:\n  train: {train_path}\n  val:   {val_path}")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print("\n1. Evaluating Stratified 5-Fold CV (OOF)")
    cv_auc = evaluate_stratified_cv(train_df, n_splits=5)
    print(f"Stratified 5-Fold CV ROC-AUC: {cv_auc:.4f}")

    print("\n2. Training Final Model State on Full Train")
    state = train_model(train_df.copy())

    print("\n3. Evaluating Model on ds_4_val.csv")
    preds = predict(state, val_df.drop(columns=[TARGET]).copy())

    assert isinstance(preds, np.ndarray), "predict() має повертати np.ndarray"
    assert preds.shape[0] == len(val_df), "довжина передбачень != довжині eval"
    assert not np.isnan(preds).any(), "у передбаченнях є NaN"

    metric = "R²" if TASK == "regression" else "ROC-AUC"
    val_auc = _score(val_df[TARGET], preds)
    print(f"ds_{DS_ID}  {metric} on validation: {val_auc:.4f}")
    print(f"Difference (|CV - Val|): {abs(cv_auc - val_auc):.4f}")

    print("\n4. Testing Reproducibility")
    preds2 = predict(train_model(train_df.copy()), val_df.drop(columns=[TARGET]).copy())
    assert np.allclose(preds, preds2), "рішення не відтворюване — перевірте фіксацію SEED"
    print("reproducibility: OK")
