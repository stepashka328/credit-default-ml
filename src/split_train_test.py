"""ПР3. Разделение исходной части на обучающую и отложенную тестовую выборки.

Перед разделением удаляются дубликаты и недопустимые записи (функция clean),
чтобы одинаковые записи не попали одновременно в обучение и проверку.
Деление 80 / 20, стратифицированное по целевой переменной, seed = 42.
В файлы записываются исходные строки data/initial.csv без изменений.
Тестовая выборка формируется один раз и используется только в ПР4.

Запуск из корня репозитория:
    python src/split_train_test.py
Результат: data/train.csv, data/test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR, ID_COL, INITIAL_PATH, SEED, TARGET
from features import clean

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TEST_SHARE = 0.2
CSV_OPTS = dict(index=False, lineterminator="\r\n")


def main() -> None:
    initial = pd.read_csv(INITIAL_PATH)
    kept = initial[initial[ID_COL].isin(clean(initial)[ID_COL])]
    train, test = train_test_split(kept, test_size=TEST_SHARE,
                                   stratify=kept[TARGET], random_state=SEED)
    train.to_csv(TRAIN_PATH, **CSV_OPTS)
    test.to_csv(TEST_PATH, **CSV_OPTS)

    print(f"Исходная часть: {len(initial)} записей, после очистки {len(kept)} "
          f"(удалено {len(initial) - len(kept)})")
    for name, part in [("train.csv", train), ("test.csv", test)]:
        print(f"  data/{name:<10} {len(part):>6} записей, "
              f"доля дефолтов {part[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
