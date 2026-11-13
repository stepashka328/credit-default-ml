"""ПР3. Пайплайн подготовки данных и признаков.

Реализует проект пайплайна из ПР2 (таблицы 5, 9 и 10 отчета ПР2):
  clean          - очистка без параметров: удаление дубликатов и записей
                   с недопустимыми значениями, перекодировка EDUCATION и MARRIAGE;
  add_features   - расчет новых признаков по плану признаков;
  make_pipeline  - расчет признаков, предобработка и модель в одном объекте Pipeline;
  prepare        - превращает любой файл (исходную часть, train/test или порцию)
                   в данные, готовые для модели.

Запуск из корня репозитория (таблица признаков + проверка повторяемости):
    python src/features.py data/incoming/batch_01.csv
Результат: data/processed/<имя файла>_features.csv
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import (BILL_COLS, DATA_DIR, ID_COL, PAY_AMT_COLS, PAY_COLS, ROOT,
                    TARGET)

PROCESSED_DIR = DATA_DIR / "processed"

# Исходные признаки модели (ID и целевая переменная в их число не входят)
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] \
    + PAY_COLS + BILL_COLS + PAY_AMT_COLS

# Новые признаки по плану признаков ПР2 (таблица 10)
UTIL_COLS = [f"util_ratio_{i}" for i in range(1, 7)]   # счет / кредитный лимит
PAYR_COLS = [f"pay_ratio_{i}" for i in range(1, 7)]    # платеж / счет
NEW_FEATURES = UTIL_COLS + PAYR_COLS + ["max_delay", "n_delays", "avg_bill"]

# Денежные суммы с тяжелым хвостом: для линейных моделей и kNN логарифмируются
MONEY_COLS = ["LIMIT_BAL"] + BILL_COLS + PAY_AMT_COLS + ["avg_bill"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка по требованиям к качеству данных (ПР2, таблица 5).

    Не имеет параметров, оцениваемых по данным, поэтому одинаково
    применяется к обучающим данным и к новым порциям.
    """
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COL]).copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    valid = (df[RAW_FEATURES].notna().all(axis=1)
             & (df["LIMIT_BAL"] > 0)
             & df["AGE"].between(18, 100)
             & df[PAY_COLS].isin(range(-2, 9)).all(axis=1)
             & (df[PAY_AMT_COLS] >= 0).all(axis=1))
    return df[valid].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Новые признаки по плану ПР2 (таблица 10): 15 столбцов пяти групп."""
    df = df.copy()
    bill = df[BILL_COLS].to_numpy(dtype=float)
    paid = df[PAY_AMT_COLS].to_numpy(dtype=float)
    limit = df["LIMIT_BAL"].to_numpy(dtype=float)[:, None]
    # Загрузка лимита: доля лимита, занятая счетом
    df[UTIL_COLS] = bill / limit
    # Доля погашения счета; при счете <= 0 платить нечего - доля равна 1;
    # платеж больше счета означает полное погашение, поэтому доля ограничена 1
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bill > 0, paid / bill, 1.0)
    df[PAYR_COLS] = np.clip(ratio, 0.0, 1.0)
    delays = df[PAY_COLS].clip(lower=0)            # -2, -1, 0 - просрочки нет
    df["max_delay"] = delays.max(axis=1)           # глубина просрочки, месяцев
    df["n_delays"] = (delays > 0).sum(axis=1)      # число месяцев с просрочкой
    df["avg_bill"] = bill.mean(axis=1)             # средний счет за полгода
    return df


def signed_log(x):
    """Логарифм с сохранением знака: суммы счетов бывают отрицательными."""
    return np.sign(x) * np.log1p(np.abs(x))


def feature_columns(features: str = "extended") -> list[str]:
    """Состав признаков: base - 23 исходных, extended - исходные и 15 новых."""
    return RAW_FEATURES + (NEW_FEATURES if features == "extended" else [])


def make_pipeline(model, kind: str = "linear", features: str = "extended") -> Pipeline:
    """Расчет признаков, предобработка и модель в одном объекте Pipeline.

    kind="linear" - для логистической регрессии, kNN и наивного Байеса:
        денежные суммы логарифмируются и стандартизуются, остальные числа
        стандартизуются, категории и статусы погашения PAY_ кодируются
        one-hot (связь PAY_1 с дефолтом немонотонна, ПР2 таблица 3);
    kind="tree" - для деревьев и ансамблей: признаки передаются как есть,
        масштабирование и кодирование деревьям не нужны.
    Параметры предобработки оцениваются только на обучающих данных при fit,
    поэтому при кросс-валидации утечки через предобработку нет.
    """
    cols = feature_columns(features)
    if kind == "linear":
        money = [c for c in cols if c in MONEY_COLS]
        cats = CAT_COLS + PAY_COLS
        other = [c for c in cols if c not in money and c not in cats]
        prep = ColumnTransformer([
            ("money", Pipeline([("log", FunctionTransformer(signed_log,
                                                           feature_names_out="one-to-one")),
                                ("scale", StandardScaler())]), money),
            ("num", StandardScaler(), other),
            # редкие значения (менее 10 записей) объединяются в одну категорию,
            # неизвестные в новой порции значения не вызывают ошибку
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=10, sparse_output=False), cats),
        ], verbose_feature_names_out=False)
    elif kind == "tree":
        prep = ColumnTransformer([("pass", "passthrough", cols)],
                                 verbose_feature_names_out=False)
    else:
        raise ValueError(f"неизвестный вид предобработки: {kind}")
    steps = []
    if features == "extended":
        steps.append(("features", FunctionTransformer(add_features)))
    steps += [("prep", prep), ("model", model)]
    return Pipeline(steps)


def prepare(path) -> tuple[pd.DataFrame, pd.Series | None]:
    """Любой файл набора -> (X, y): очищенные исходные признаки и целевая.

    X подается на вход объекта Pipeline, который сам рассчитывает новые
    признаки. Для порции без целевой переменной y = None.
    """
    df = clean(pd.read_csv(path))
    y = df[TARGET] if TARGET in df.columns else None
    return df[RAW_FEATURES], y


def build_feature_table(path) -> pd.DataFrame:
    """Таблица признаков файла: ID, 38 признаков и целевая (если есть)."""
    df = clean(pd.read_csv(path))
    keep = [ID_COL] + feature_columns("extended") \
        + ([TARGET] if TARGET in df.columns else [])
    return add_features(df)[keep]


def _md5(frame: pd.DataFrame) -> str:
    return hashlib.md5(frame.to_csv(index=False).encode()).hexdigest()


def main(path: str) -> None:
    path = Path(path)
    raw_n = len(pd.read_csv(path))
    first = build_feature_table(path)
    second = build_feature_table(path)
    print(f"Файл {path.as_posix()}: {raw_n} записей, после очистки {len(first)}, "
          f"признаков {first.shape[1] - 1 - (TARGET in first.columns)}")

    # Повторяемость: два запуска на одних данных дают одинаковый результат
    same_table = first.equals(second)
    X, y = prepare(path)
    prep = make_pipeline(None, kind="linear")[:-1]   # пайплайн без модели
    m1 = prep.fit_transform(X)
    m2 = make_pipeline(None, kind="linear")[:-1].fit_transform(X)
    same_matrix = np.array_equal(m1, m2)
    print(f"Повторный расчет таблицы признаков совпадает: {same_table} "
          f"(MD5 {_md5(first)} / {_md5(second)})")
    print(f"Повторное обучение предобработки дает ту же матрицу: {same_matrix} "
          f"(размер {m1.shape[0]} x {m1.shape[1]})")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{path.stem}_features.csv"
    first.to_csv(out, index=False)
    print(f"Сохранено: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/initial.csv")
