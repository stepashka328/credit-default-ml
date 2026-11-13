"""ПР2. Разведочный анализ исходной части данных (data/initial.csv).

Состав анализа — по п. 2.2.2 методички: структура набора, целевая
переменная, качество данных, связь признаков с целевой, взаимосвязи
признаков и проверка предметных гипотез (хи-квадрат, Манна – Уитни).

Запуск из корня репозитория:
    python src/eda.py

Результаты: reports/pr2/ (таблицы *.csv, графики *.png, eda_output.txt).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from config import (BILL_COLS, ID_COL, INITIAL_PATH, PAY_AMT_COLS, PAY_COLS,
                    REPORTS_DIR, TARGET)

OUT = REPORTS_DIR / "pr2"
_lines: list[str] = []


def log(*parts) -> None:
    """Печатает строку и сохраняет её для файла eda_output.txt."""
    line = " ".join(str(p) for p in parts)
    print(line)
    _lines.append(line)


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INITIAL_PATH)
    y = df[TARGET]

    log("== СТРУКТУРА ==")
    log("записей и столбцов:", df.shape)
    pd.DataFrame({"dtype": df.dtypes.astype(str), "nunique": df.nunique(),
                  "min": df.min(), "max": df.max()}).to_csv(OUT / "structure.csv")

    log("== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ==")
    log("число по классам:", y.value_counts().to_dict())
    log("доли по классам:", y.value_counts(normalize=True).round(3).to_dict())

    log("== КАЧЕСТВО ДАННЫХ ==")
    log("пропуски:", int(df.isna().sum().sum()))
    log("дубли ID:", int(df[ID_COL].duplicated().sum()),
        "| дубли строк без ID:", int(df.drop(columns=ID_COL).duplicated().sum()))
    log("EDUCATION вне 1-4:", int((~df.EDUCATION.isin([1, 2, 3, 4])).sum()),
        "| значения:", df.EDUCATION.value_counts().to_dict())
    log("MARRIAGE вне 1-3:", int((~df.MARRIAGE.isin([1, 2, 3])).sum()))
    log("LIMIT_BAL <= 0:", int((df.LIMIT_BAL <= 0).sum()),
        "| AGE вне 18-100:", int((~df.AGE.between(18, 100)).sum()))
    log("PAY_AMT < 0:", int((df[PAY_AMT_COLS] < 0).sum().sum()),
        "| BILL_AMT < 0:", int((df[BILL_COLS] < 0).sum().sum()))
    log("PAY вне -2..8:", int((~df[PAY_COLS].isin(range(-2, 9))).sum().sum()))
    for c in ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        share = ((df[c] > q3 + 1.5 * iqr) | (df[c] < q1 - 1.5 * iqr)).mean()
        log(f"выбросы по IQR {c}: {share:.1%}")

    log("== ДОЛЯ ДЕФОЛТОВ ПО PAY_1 ==")
    rate = df.groupby("PAY_1")[TARGET].agg(["mean", "count"]).round(3)
    log(rate.to_string())
    rate.to_csv(OUT / "default_by_pay1.csv")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    df.groupby("PAY_1")[TARGET].mean().plot.bar(ax=ax[0], color="#6B9DAD")
    ax[0].set(title="Доля дефолтов по PAY_1",
              xlabel="Статус погашения за сентябрь (PAY_1)", ylabel="Доля дефолтов")
    ax[1].boxplot([df.loc[y == 0, "LIMIT_BAL"], df.loc[y == 1, "LIMIT_BAL"]])
    ax[1].set_xticks([1, 2], ["нет дефолта", "дефолт"])
    ax[1].set(title="Кредитный лимит по классам", ylabel="LIMIT_BAL, NT$")
    ax[2].hist(df.AGE, bins=30, color="#6B9DAD")
    ax[2].set(title="Распределение возраста", xlabel="Возраст, лет",
              ylabel="Число клиентов")
    plt.tight_layout()
    plt.savefig(OUT / "fig01_distributions.png", dpi=150)
    plt.close()

    corr = df[["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
              + [TARGET]].corr()
    corr.to_csv(OUT / "corr.csv")
    log("== КОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ (8 наибольших по модулю) ==")
    top = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(8)
    log(top.round(3).to_string())
    log("корреляция BILL_AMT1 - BILL_AMT2:", round(corr.loc["BILL_AMT1", "BILL_AMT2"], 3))
    plt.figure(figsize=(9, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr)), corr.columns, fontsize=7)
    plt.colorbar(label="Коэффициент корреляции Пирсона")
    plt.title("Корреляционная матрица числовых признаков")
    plt.tight_layout()
    plt.savefig(OUT / "fig02_corr.png", dpi=150)
    plt.close()

    log("== ПРОВЕРКА ГИПОТЕЗ ==")
    chi, p, *_ = stats.chi2_contingency(pd.crosstab(df.PAY_1, y))
    log(f"H1 доля дефолтов зависит от PAY_1: хи-квадрат = {chi:.1f}, {fmt_p(p)}")
    a, b = df.loc[y == 1, "LIMIT_BAL"], df.loc[y == 0, "LIMIT_BAL"]
    p = stats.mannwhitneyu(a, b).pvalue
    log(f"H2 LIMIT_BAL: медиана дефолт = {a.median():.0f}, "
        f"нет дефолта = {b.median():.0f}, Манна - Уитни {fmt_p(p)}")
    d = df[df.BILL_AMT1 > 0]
    r = d.PAY_AMT1 / d.BILL_AMT1
    a, b = r[d[TARGET] == 1], r[d[TARGET] == 0]
    p = stats.mannwhitneyu(a, b).pvalue
    log(f"H3 PAY_AMT1/BILL_AMT1 (BILL_AMT1 > 0): медиана дефолт = {a.median():.3f}, "
        f"нет дефолта = {b.median():.3f}, Манна - Уитни {fmt_p(p)}")

    (OUT / "eda_output.txt").write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
