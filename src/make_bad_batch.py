"""ПР4. Порча порции для демонстрации проверок качества (Q1–Q10).

Из batch_01.csv создается файл data/demo/bad_batch.csv из 300 записей,
в который вносится полный набор нарушений из ПР2 (таблица 5):
два одинаковых ID, пропуски, отрицательный лимит, невозможный возраст,
неописанный код образования, недопустимый статус PAY_1, отрицательный
платёж и полные дубликаты строк. Файл создается скриптом, поэтому порча
воспроизводится и не затрагивает исходные данные.

Запуск из корня репозитория:
    python src/make_bad_batch.py
Результат: data/demo/bad_batch.csv (300 записей)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, ID_COL, SEED, TARGET

OUT_DIR = DATA_DIR / "demo"
OUT_PATH = OUT_DIR / "bad_batch.csv"


def make_bad_batch() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "incoming" / "batch_01.csv")
    _, bad = train_test_split(df, train_size=2700, stratify=df[TARGET],
                              random_state=SEED)
    bad = bad.head(300).reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    bad.loc[0, ID_COL] = bad.loc[1, ID_COL]                 # Q1: одинаковый ID
    bad.loc[2:5, "LIMIT_BAL"] = np.nan                      # Q2: пропуски
    bad.loc[6, "LIMIT_BAL"] = -1                            # Q3: лимит <= 0
    bad.loc[7, "AGE"] = 150                                 # Q4: возраст вне 18..100
    bad.loc[8:11, "EDUCATION"] = 9                          # Q5: код вне 0..6
    bad.loc[12, "MARRIAGE"] = 7                             # Q6: код вне 0..3
    bad.loc[13, "PAY_1"] = 12                               # Q7: статус вне -2..8
    bad.loc[14:16, "PAY_AMT1"] = -5                         # Q8: отрицательный платёж
    bad.loc[17] = bad.loc[18]                               # Q9: полный дубликат строки
    bad.loc[19:24, "AGE"] = np.nan                          # Q2: пропуски в другом поле
    bad.loc[25, "BILL_AMT1"] = float("nan")                 # Q2: пропуск в счёте
    return bad


def main() -> None:
    bad = make_bad_batch()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bad.to_csv(OUT_PATH, index=False)
    print(f"Испорченная порция: {len(bad)} записей -> "
          f"{OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
