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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score

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
DS_ID = 1                        # номер вашого варіанту -> ds_1, ds_2, ...
TARGET = "total_UPDRS"           # цільовий стовпчик вашого варіанту
TASK = "regression"              # "regression" | "classification"

DATA_DIR = os.path.join("out", "for_students")


# ---------------------------------------------------------------------------
# Контракт рішення
# ---------------------------------------------------------------------------
def train_model(train_df: pd.DataFrame):
    """Отримує train (з таргетом). Робить весь препроцесинг + навчання.
    Повертає state (модель / словник / кортеж) для передачі в predict()."""
    feature_cols = [c for c in train_df.columns if c != TARGET]

    # мінімальний препроцесинг: тільки числові ознаки, NaN -> медіана
    X = train_df[feature_cols].select_dtypes(include=[np.number]).copy()
    used_cols = list(X.columns)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = train_df[TARGET]

    if TASK == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(X, y)

    return {"model": model, "used_cols": used_cols, "medians": medians}


def predict(state, eval_df: pd.DataFrame) -> np.ndarray:
    """Отримує state і eval-набір БЕЗ таргету. Повертає 1-D np.ndarray.
    Для класифікації — ймовірність позитивного класу (для ROC-AUC)."""
    X = eval_df[state["used_cols"]].copy()
    X = X.fillna(state["medians"])
    model = state["model"]

    if TASK == "classification":
        return np.asarray(model.predict_proba(X)[:, 1])
    return np.asarray(model.predict(X))


# ---------------------------------------------------------------------------
# Локальна перевірка: чи відповідає розв'язок контракту + метрика на валідації
# ---------------------------------------------------------------------------
def _score(y_true, y_pred):
    if TASK == "regression":
        return r2_score(y_true, y_pred)   # R²
    return roc_auc_score(y_true, y_pred)


if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"ds_{DS_ID}_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, f"ds_{DS_ID}_val.csv"))

    state = train_model(train_df.copy())
    preds = predict(state, val_df.drop(columns=[TARGET]).copy())

    assert isinstance(preds, np.ndarray), "predict() має повертати np.ndarray"
    assert preds.shape[0] == len(val_df), "довжина передбачень != довжині eval"
    assert not np.isnan(preds).any(), "у передбаченнях є NaN"

    metric = "R²" if TASK == "regression" else "ROC-AUC"
    print(f"ds_{DS_ID}  {metric} on validation: {_score(val_df[TARGET], preds):.4f}")

    # перевірка відтворюваності
    preds2 = predict(train_model(train_df.copy()), val_df.drop(columns=[TARGET]).copy())
    assert np.allclose(preds, preds2), "рішення не відтворюване — перевірте фіксацію SEED"
    print("reproducibility: OK")
