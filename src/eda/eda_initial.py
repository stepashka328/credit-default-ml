import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

df = pd.read_csv("data/initial.csv")
T = df.columns[-1]
print("целевая:", T, "| столбцы:", list(df.columns))
Path("reports/tables").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)
pay = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
amt = [f"PAY_AMT{i}" for i in range(1, 7)]

print("== СТРУКТУРА ==", df.shape)
pd.DataFrame({"dtype": df.dtypes.astype(str), "nunique": df.nunique(),
              "min": df.min(), "max": df.max()}).to_csv("reports/tables/structure.csv")

print("== ЦЕЛЕВАЯ ==")
print(df[T].value_counts(), df[T].value_counts(normalize=True).round(3))

print("== КАЧЕСТВО ==")
print("пропуски:", int(df.isna().sum().sum()))
print("дубли ID:", int(df["ID"].duplicated().sum()),
      "| дубли строк без ID:", int(df.drop(columns="ID").duplicated().sum()))
print("EDUCATION вне 1-4:", int((~df.EDUCATION.isin([1, 2, 3, 4])).sum()),
      "| значения:", df.EDUCATION.value_counts().to_dict())
print("MARRIAGE вне 1-3:", int((~df.MARRIAGE.isin([1, 2, 3])).sum()))
print("LIMIT_BAL<=0:", int((df.LIMIT_BAL <= 0).sum()),
      "| AGE вне 18-100:", int((~df.AGE.between(18, 100)).sum()))
print("PAY_AMT<0:", int((df[amt] < 0).sum().sum()),
      "| BILL_AMT<0:", int((df[bill] < 0).sum().sum()))
print("PAY вне -2..8:", int((~df[pay].isin(range(-2, 9))).sum().sum()))
for c in ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    print(f"выбросы IQR {c}: {((df[c] > q3 + 1.5*iqr) | (df[c] < q1 - 1.5*iqr)).mean():.1%}")

print("== ДОЛЯ ДЕФОЛТА ПО PAY_1 ==")
rate = df.groupby("PAY_1")[T].agg(["mean", "count"]).round(3)
print(rate); rate.to_csv("reports/tables/default_by_pay0.csv")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df.groupby("PAY_1")[T].mean().plot.bar(ax=ax[0], title="Доля дефолта по PAY_1")
df.boxplot("LIMIT_BAL", by=T, ax=ax[1]); ax[1].set_title("LIMIT_BAL по классам")
df.AGE.hist(bins=30, ax=ax[2]); ax[2].set_title("AGE")
plt.suptitle(""); plt.tight_layout(); plt.savefig("reports/figures/fig01_distributions.png", dpi=150)

corr = df[["LIMIT_BAL", "AGE"] + pay + bill + amt + [T]].corr()
corr.to_csv("reports/tables/corr.csv")
print("== КОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ ==")
print(corr[T].drop(T).sort_values(key=abs, ascending=False).head(8).round(3))
print("корр. BILL_AMT1-BILL_AMT2:", round(corr.loc["BILL_AMT1", "BILL_AMT2"], 3))
plt.figure(figsize=(9, 8)); plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=6)
plt.yticks(range(len(corr)), corr.columns, fontsize=6); plt.colorbar()
plt.tight_layout(); plt.savefig("reports/figures/fig02_corr.png", dpi=150)

print("== ГИПОТЕЗЫ ==")
chi, p, *_ = stats.chi2_contingency(pd.crosstab(df.PAY_1, df[T]))
print(f"H1 PAY_1 vs дефолт: chi2={chi:.1f}, p={p:.2e}")
a, b = df.loc[df[T] == 1, "LIMIT_BAL"], df.loc[df[T] == 0, "LIMIT_BAL"]
print(f"H2 LIMIT_BAL: медиана дефолт={a.median():.0f}, не дефолт={b.median():.0f}, p={stats.mannwhitneyu(a, b).pvalue:.2e}")
d = df[df.BILL_AMT1 > 0].copy(); d["r"] = d.PAY_AMT1 / d.BILL_AMT1
a, b = d.loc[d[T] == 1, "r"], d.loc[d[T] == 0, "r"]
print(f"H3 PAY_AMT1/BILL_AMT1: медиана дефолт={a.median():.3f}, не дефолт={b.median():.3f}, p={stats.mannwhitneyu(a, b).pvalue:.2e}")
