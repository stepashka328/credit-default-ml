"""Тесты пайплайна подготовки данных и признаков (src/features.py).

Запуск из корня репозитория:
    python -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import INCOMING_DIR  # noqa: E402
from features import (NEW_FEATURES, RAW_FEATURES, add_features, clean,  # noqa: E402
                      make_pipeline, prepare)
from sklearn.linear_model import LogisticRegression  # noqa: E402


def client(**overrides) -> dict:
    row = {"ID": 1, "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1,
           "AGE": 30, "dpnm": 0}
    row.update({f"PAY_{i}": 0 for i in range(1, 7)})
    row.update({f"BILL_AMT{i}": 20000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 2000 for i in range(1, 7)})
    row.update(overrides)
    return row


@pytest.fixture
def batch() -> pd.DataFrame:
    return pd.read_csv(INCOMING_DIR / "batch_01.csv")


def test_clean_recodes_and_drops():
    df = pd.DataFrame([
        client(ID=1, EDUCATION=5, MARRIAGE=0),
        client(ID=2, EDUCATION=5, MARRIAGE=0),   # дубликат строки 1 (кроме ID)
        client(ID=3, AGE=150),                   # недопустимый возраст
        client(ID=4, PAY_AMT1=-5),               # отрицательный платеж
    ])
    out = clean(df)
    assert list(out.ID) == [1]
    assert out.EDUCATION.iloc[0] == 4 and out.MARRIAGE.iloc[0] == 3


def test_add_features_values():
    df = pd.DataFrame([client(BILL_AMT1=50000, PAY_AMT1=10000, BILL_AMT2=0,
                              PAY_1=2, PAY_2=1, PAY_3=-1)])
    f = add_features(df)
    assert set(NEW_FEATURES) <= set(f.columns)
    assert f.util_ratio_1.iloc[0] == pytest.approx(0.5)
    assert f.pay_ratio_1.iloc[0] == pytest.approx(0.2)
    assert f.pay_ratio_2.iloc[0] == 1.0          # счета нет - доля погашения 1
    assert f.max_delay.iloc[0] == 2 and f.n_delays.iloc[0] == 2


def test_prepare_repeatable(batch, tmp_path):
    path = tmp_path / "batch.csv"
    batch.to_csv(path, index=False)
    (x1, y1), (x2, y2) = prepare(path), prepare(path)
    assert list(x1.columns) == RAW_FEATURES
    pd.testing.assert_frame_equal(x1, x2)
    m1 = make_pipeline(None)[:-1].fit_transform(x1)
    m2 = make_pipeline(None)[:-1].fit_transform(x2)
    assert np.array_equal(m1, m2)


def test_pipeline_handles_unseen_category(batch):
    X, y = clean(batch)[RAW_FEATURES], clean(batch)["dpnm"]
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    new = pd.DataFrame([client(PAY_1=8, PAY_2=8)])[RAW_FEATURES]  # редкие коды
    proba = model.predict_proba(new)[:, 1]
    assert proba.shape == (1,) and 0.0 <= proba[0] <= 1.0
