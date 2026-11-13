"""ПР2. Разделение исходного набора на исходную часть и порции «новых» данных.

В наборе нет даты, поэтому деление случайное (методичка, п. 2.2.1):
исходная часть — 60 % записей, остаток — четыре порции по 3 000 записей.
Деление стратифицировано по целевой переменной, начальное значение
генератора фиксировано. Исходный файл data/raw не изменяется.

Запуск из корня репозитория:
    python src/split_batches.py
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import INCOMING_DIR, INITIAL_PATH, RAW_PATH, SEED, TARGET

N_BATCHES = 4
INITIAL_SHARE = 0.6
# Концы строк заданы явно: файлы побайтно совпадают на любой ОС
# (и с контрольными суммами, зафиксированными при сдаче ПР2).
CSV_OPTS = dict(index=False, lineterminator="\r\n")


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    initial, rest = train_test_split(
        df, train_size=INITIAL_SHARE, stratify=df[TARGET], random_state=SEED)
    initial.to_csv(INITIAL_PATH, **CSV_OPTS)

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    rows = [("initial.csv", len(initial), initial[TARGET].mean())]
    skf = StratifiedKFold(n_splits=N_BATCHES, shuffle=True, random_state=SEED)
    for i, (_, idx) in enumerate(skf.split(rest, rest[TARGET]), start=1):
        batch = rest.iloc[idx]
        name = f"batch_{i:02d}.csv"
        batch.to_csv(INCOMING_DIR / name, **CSV_OPTS)
        rows.append((f"incoming/{name}", len(batch), batch[TARGET].mean()))

    print(f"Исходный набор: {len(df)} записей, доля дефолтов {df[TARGET].mean():.3f}")
    for name, n, rate in rows:
        print(f"  data/{name:<22} {n:>6} записей, доля дефолтов {rate:.3f}")


if __name__ == "__main__":
    main()
