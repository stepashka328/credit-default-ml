"""ПР4. Стресс-порция batch_05: имитация дрейфа данных для демонстрации
работы мониторинга.

Даты в наборе нет, поэтому порции batch_01–batch_04 (случайное
стратифицированное разбиение) статистически неотличимы от исходной части:
PSI ключевых признаков на них близок к нулю, и работа правил мониторинга
не может быть показана. Пособие (п. 2.2.1) разрешает в этом случае
искусственно изменить одну из порций с обязательным описанием этого в
отчете.

Стресс-порция имитирует ухудшение экономических условий: у половины
клиентов batch_04 статусы погашения сдвигаются на один месяц в сторону
просрочки, счета вырастают на 30 %, платежи падают на 20 %. Целевая
переменная не пересчитывается: факт дефолта остаётся фактом, поэтому
качество модели на стресс-порции не искажается выдуманной разметкой.
Записи batch_05 - это те же клиенты batch_04 с изменёнными признаками,
поэтому в ПР5 при переобучении порция batch_04 в обучение претендента не
включается (иначе те же клиенты встретились бы в обучении и проверке).

Запуск из корня репозитория:
    python src/make_stress_batch.py
Результат: data/incoming/batch_05.csv (3000 записей)
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, BILL_COLS, PAY_AMT_COLS, PAY_COLS, SEED, TARGET

OUT_PATH = DATA_DIR / "incoming" / "batch_05.csv"
AFFECTED_SHARE = 0.5      # доля клиентов, затронутых ухудшением условий
BILL_X = 1.3              # счета выросли на 30 %
PAY_X = 0.8               # платежи упали на 20 %


def make_stress_batch() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "incoming" / "batch_04.csv").copy()
    rng = np.random.default_rng(SEED)
    affected = rng.random(len(df)) < AFFECTED_SHARE
    df.loc[affected, PAY_COLS] = (df.loc[affected, PAY_COLS] + 1).clip(-2, 8)
    df.loc[affected, BILL_COLS] = (df.loc[affected, BILL_COLS] * BILL_X).round(0)
    df.loc[affected, PAY_AMT_COLS] = (df.loc[affected, PAY_AMT_COLS] * PAY_X).round(0)
    return df


def main() -> None:
    df = make_stress_batch()
    df.to_csv(OUT_PATH, index=False, lineterminator="\r\n")
    src = pd.read_csv(DATA_DIR / "incoming" / "batch_04.csv")
    print(f"Стресс-порция: {len(df)} записей -> {OUT_PATH.as_posix()}")
    print(f"  доля дефолтов: {df[TARGET].mean():.3f} (в batch_04: {src[TARGET].mean():.3f})")
    print(f"  PAY_1 >= 1: {(df.PAY_1 >= 1).mean():.1%} "
          f"(в batch_04: {(src.PAY_1 >= 1).mean():.1%})")
    print(f"  средний счёт: {df.BILL_AMT1.mean():.0f} "
          f"(в batch_04: {src.BILL_AMT1.mean():.0f})")


if __name__ == "__main__":
    main()
