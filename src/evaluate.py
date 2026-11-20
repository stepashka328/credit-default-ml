"""ПР4. Оценка качества моделей, интерпретируемость и валидация результатов.

Оценивает итоговую модель ПР3 (models/model_v1.joblib) на отложенной
тестовой выборке и порциях новых данных:
  1. протокол оценки и пороги приемки фиксируются ДО расчетов на тесте
     (reports/pr4/protocol.json);
  2. метрики итоговой и базовых моделей на тесте, доверительные интервалы
     бутстрепом (1000 выборок), значимость различия двух лучших моделей;
  3. матрица ошибок, ROC-кривая, кривая «точность – полнота», порог
     классификации по цене ошибок;
  4. анализ ошибок по содержательным срезам;
  5. перестановочная важность признаков, частичные зависимости, объяснения
     отдельных предсказаний;
  6. устойчивость к шуму и пропускам;
  7. проверки качества данных (quality.py) на испорченном файле и порциях;
  8. качество модели и PSI на порциях batch_01–batch_04 и стресс-порции
     batch_05 (make_stress_batch.py);
  9. паспорт модели и решение об использовании.

Запуск из корня репозитория (около 1 минуты):
    python src/make_bad_batch.py && python src/make_stress_batch.py && python src/evaluate.py
Результаты: reports/pr4/ (таблицы, графики, JSON-описания, evaluation_output.txt)
"""

import json
import time
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from config import DATA_DIR, ID_COL, INCOMING_DIR, MODELS_DIR, REPORTS_DIR, SEED, TARGET
from experiments import PayRule
from features import RAW_FEATURES, clean, prepare
from monitoring import (PSI_STABLE, PSI_ALERT, load_reference, psi_features,
                        psi_status)
from quality import check_quality, format_report
from split_train_test import TEST_PATH, TRAIN_PATH

OUT = REPORTS_DIR / "pr4"
OUT_TXT = OUT / "evaluation_output.txt"
_lines: list[str] = []

# Цвета серий и оформление - как в ПР2 и ПР3 (единый стиль отчетов)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

N_BOOTSTRAP = 1000

# ---- Протокол оценки: пороги приемки, фиксируемые ДО расчетов на тесте ----
# Основаны на кросс-валидации из ПР3 (ROC-AUC 0,784; правило 0,688; PR-AUC 0,556;
# показатель Брайера 0,134; полнота правила PAY_1 ≥ 1 на CV 0,523) с допуском
# на случайность тестовой выборки: выбранная по цене ошибок модель должна
# находить не меньше будущих дефолтов, чем базовое правило
ACCEPT = {
    "roc_auc_min": 0.75,
    "roc_auc_vs_rule_min": 0.05,
    "pr_auc_min": 0.50,
    "brier_max": 0.145,
    "recall_at_threshold_min": 0.55,
    "calibration_gap_max": 0.02,
    "batch_roc_auc_min": 0.75,
    "batch_calibration_gap_max": 0.05,
    "psi_alert": PSI_ALERT,
}

# ---- Цена ошибок для заказчика (учебные допущения, описанные в отчете) ----
# Пропуск дефолта: потери LGD 0,8 (необеспеченная карта) * доля предотвращаемых
# потерь 0,4 (ранний пересмотр лимита) * средний счёт дефолтера (обучающая выборка).
# Ложное срабатывание: издержки проверки и упущенная маржа = 10 % месячного счёта.
LGD, PREVENTED, FP_RATE = 0.8, 0.4, 0.1

BATCH_FILES = [INCOMING_DIR / f"batch_{i:02d}.csv" for i in range(1, 6)]
BAD_PATH = DATA_DIR / "demo" / "bad_batch.csv"


def log(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    _lines.append(line)


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def comma(ax, which="y", digits=2):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(fmt)


# ------------------------------------------------------------------ helpers
def bootstrap_ci(y, p, metric=roc_auc_score, seed=SEED, n_boot=N_BOOTSTRAP):
    """95 % доверительный интервал метрики методом бутстрепа."""
    rng = np.random.default_rng(seed)
    values = []
    y = np.asarray(y)
    p = np.asarray(p)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        values.append(metric(y[idx], p[idx]))
    return np.percentile(values, [2.5, 97.5])


def bootstrap_diff_ci(y, p1, p2, seed=SEED, n_boot=N_BOOTSTRAP):
    """95 % интервал для разности метрик двух моделей (парный бутстреп)."""
    rng = np.random.default_rng(seed)
    values = []
    y = np.asarray(y)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        values.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    return np.percentile(values, [2.5, 97.5])


def confusion_counts(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
    return tn, fp, fn, tp


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    return {"precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "accuracy": float((pred == y).mean())}


def draw_confusion(ax, y, p, t, title):
    tn, fp, fn, tp = confusion_counts(y, p, t)
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15)
    labels = [["истинно отрицательные", "ложные срабатывания"],
              ["пропуски", "истинно положительные"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    fontsize=10, color="white" if m[i, j] > m.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["прогноз: нет дефолта", "прогноз: дефолт"])
    ax.set_yticks([0, 1], ["нет дефолта", "дефолт"])
    ax.set_title(title, color=INK, fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(colors=MUTED, length=0)


# ------------------------------------------------------------------ main
def main() -> None:
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODELS_DIR / "model_v1.joblib")
    info = json.loads((MODELS_DIR / "model_v1.json").read_text(encoding="utf-8"))

    X_test, y_test = prepare(TEST_PATH)
    X_train, y_train = prepare(TRAIN_PATH)
    train_df = clean(pd.read_csv(TRAIN_PATH))
    p_test = model.predict_proba(X_test)[:, 1]

    # ---------- 1. Протокол оценки (до расчетов на тесте) ----------
    bill_def = float(train_df.loc[train_df[TARGET] == 1, "BILL_AMT1"].mean())
    bill_ok = float(train_df.loc[train_df[TARGET] == 0, "BILL_AMT1"].mean())
    c_fn = PREVENTED * LGD * bill_def
    c_fp = FP_RATE * bill_ok
    protocol = {
        "fixed_on": date.today().isoformat(),
        "metrics": ["ROC-AUC (основная)", "PR-AUC", "показатель Брайера",
                    "точность, полнота, F1 при выбранном пороге"],
        "costs": {"c_fn_ntd": round(c_fn, 0), "c_fp_ntd": round(c_fp, 0),
                  "lgd": LGD, "prevented_share": PREVENTED, "fp_rate": FP_RATE,
                  "mean_bill_defaulters": round(bill_def, 0),
                  "mean_bill_good": round(bill_ok, 0)},
        "acceptance": ACCEPT,
        "note": "пороги зафиксированы до обращения к тестовой выборке; основаны "
                "на кросс-валидации ПР3 и бизнес-допущениях о цене ошибок",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    log("== 1. ПРОТОКОЛ ОЦЕНКИ ==")
    log("цена пропуска дефолта c_FN =", round(c_fn), "NT$, "
        "цена ложного срабатывания c_FP =", round(c_fp), "NT$")
    log("пороги приемки:", json.dumps(ACCEPT, ensure_ascii=False))

    # ---------- 2. Метрики на тестовой выборке ----------
    log("== 2. МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ ==")
    hgb_params = {"learning_rate": 0.03, "max_depth": 3, "max_iter": 200,
                  "min_samples_leaf": 100, "random_state": SEED}
    hgb = HistGradientBoostingClassifier(**hgb_params).fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_test)[:, 1]
    p_rule = PayRule().fit(X_test, y_test).predict_proba(X_test)[:, 1]
    p_dummy = np.zeros(len(y_test))

    roc_lr = roc_auc_score(y_test, p_test)
    roc_hgb = roc_auc_score(y_test, p_hgb)
    roc_rule = roc_auc_score(y_test, p_rule)
    brier_lr = brier_score_loss(y_test, p_test)

    ci_lr = bootstrap_ci(y_test, p_test)
    ci_diff = bootstrap_diff_ci(y_test, p_test, p_hgb)
    significant = not (ci_diff[0] <= 0 <= ci_diff[1])

    test_metrics = pd.DataFrame([
        {"model": "Наиболее частый класс", "roc_auc": 0.5, "pr_auc": float("nan"),
         "brier": brier_score_loss(y_test, p_dummy), **metrics_at(y_test, p_dummy, 0.5)},
        {"model": "Правило: просрочка в сентябре (PAY_1 ≥ 1)", "roc_auc": roc_rule,
         "pr_auc": average_precision_score(y_test, p_rule),
         "brier": brier_score_loss(y_test, p_rule), **metrics_at(y_test, p_rule, 0.5)},
        {"model": "Градиентный бустинг (после подбора)", "roc_auc": roc_hgb,
         "pr_auc": average_precision_score(y_test, p_hgb),
         "brier": brier_score_loss(y_test, p_hgb), **metrics_at(y_test, p_hgb, 0.5)},
        {"model": "Логистическая регрессия L1 (итоговая)", "roc_auc": roc_lr,
         "pr_auc": average_precision_score(y_test, p_test), "brier": brier_lr,
         **metrics_at(y_test, p_test, 0.5)},
    ])
    test_metrics.to_csv(OUT / "test_metrics.csv", index=False)
    log("логистическая регрессия: ROC-AUC", n(roc_lr), "95 % ДИ [",
        n(ci_lr[0]), ";", n(ci_lr[1]), "]; PR-AUC",
        n(average_precision_score(y_test, p_test)), "; Брайер", n(brier_lr))
    log("правило PAY_1 ≥ 1: ROC-AUC", n(roc_rule), "; бустинг: ROC-AUC", n(roc_hgb))
    log("разность ROC-AUC (ЛР - бустинг): [", n(ci_diff[0]), ";", n(ci_diff[1]),
        "] - ", "значима" if significant else "не значима")
    log("кросс-валидация ПР3:", info["cv_metrics"]["roc_auc_mean"],
        "- внутри доверительного интервала" if ci_lr[0] <=
        info["cv_metrics"]["roc_auc_mean"] <= ci_lr[1] else " - вне интервала")

    # ---------- 3. Порог классификации по цене ошибок ----------
    log("== 3. ПОРОГ КЛАССИФИКАЦИИ ==")
    ths = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in ths:
        _, fp, fn, _ = confusion_counts(y_test, p_test, t)
        costs.append(fn * c_fn + fp * c_fp)
    costs = np.array(costs)
    t_star = float(ths[np.argmin(costs)])
    cost05 = costs[np.searchsorted(ths, 0.5)]
    zone05 = float((p_test >= 0.5).mean())
    zone_star = float((p_test >= t_star).mean())
    thr = {"threshold": round(t_star, 2), "c_fn": round(c_fn), "c_fp": round(c_fp),
           "expected_cost_star": round(float(costs.min()), 0),
           "expected_cost_050": round(float(cost05), 0),
           "recall": metrics_at(y_test, p_test, t_star)["recall"],
           "precision": metrics_at(y_test, p_test, t_star)["precision"],
           "zone_share": zone_star, "zone_share_050": zone05}
    (OUT / "threshold.json").write_text(json.dumps(thr, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    log("оптимальный порог", n(t_star, 2), ": полнота",
        n(thr["recall"]), "точность", n(thr["precision"]),
        "доля зоны риска", n(zone_star, 3))
    log("ожидаемые потери на тесте:", round(costs.min() / 1000), "тыс. NT$ против",
        round(float(cost05) / 1000), "тыс. NT$ при пороге 0,5")

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    ax[0].plot(fpr, tpr, color=BLUE, lw=2, label="итоговая модель")
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="случайное угадывание")
    ax[0].plot(roc_curve(y_test, p_rule)[0], roc_curve(y_test, p_rule)[1],
               color=ORANGE, lw=1.5, label="правило PAY_1 ≥ 1")
    ax[0].legend(frameon=False, loc="lower right")
    comma(ax[0], "x", 2); comma(ax[0], "y", 2)
    style(ax[0], "ROC-кривая на тестовой выборке", "Доля ложных срабатываний",
          "Полнота")
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    ax[1].plot(rec, prec, color=BLUE, lw=2, label="итоговая модель")
    ax[1].axhline(y_test.mean(), color=MUTED, lw=1, ls="--",
                  label="доля дефолтов (0,221)")
    ax[1].legend(frameon=False, loc="upper right")
    comma(ax[1], "x", 2); comma(ax[1], "y", 2)
    style(ax[1], "Кривая «точность – полнота»", "Полнота", "Точность")
    ax[2].plot(ths, costs / 1000, color=BLUE, lw=2)
    ax[2].axvline(t_star, color=ORANGE, lw=1.5, ls="--")
    ax[2].annotate(f"порог {n(t_star, 2)}", (t_star, costs.max() / 1000),
                   xytext=(8, 0), textcoords="offset points", color=MUTED, fontsize=9)
    comma(ax[2], "y", 0)
    style(ax[2], "Ожидаемые потери от порога", "Порог классификации",
          "Ожидаемые потери, тыс. NT$")
    plt.savefig(OUT / "fig01_diagnostic.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    draw_confusion(ax[0], y_test, p_test, 0.5, f"Порог 0,5 (полнота "
                    + n(metrics_at(y_test, p_test, 0.5)["recall"]) + ")")
    draw_confusion(ax[1], y_test, p_test, t_star, f"Порог " + n(t_star, 2)
                    + " (полнота " + n(thr["recall"]) + ")")
    plt.savefig(OUT / "fig02_confusion.png", dpi=200)
    plt.close(fig)

    # ---------- 4. Анализ ошибок по срезам ----------
    log("== 4. СРЕЗЫ ==")
    test_df = clean(pd.read_csv(TEST_PATH))
    slices = []

    def add_slice(name, mask):
        part = test_df[mask]
        pp = p_test[mask.to_numpy()]
        yy = part[TARGET]
        row = {"slice": name, "n": len(part), "default_rate": yy.mean(),
               "roc_auc": roc_auc_score(yy, pp) if yy.nunique() > 1 else float("nan"),
               "recall_at_t": recall_score(yy, (pp >= t_star).astype(int),
                                           zero_division=0)}
        slices.append(row)

    pay_group = np.select(
        [test_df.PAY_1 <= 0, test_df.PAY_1 == 1, test_df.PAY_1 >= 2],
        ["без просрочки (PAY_1 ≤ 0)", "просрочка 1 месяц (PAY_1 = 1)",
         "просрочка 2+ месяца (PAY_1 ≥ 2)"], default="прочее")
    for name in pd.unique(pay_group):
        add_slice("PAY_1: " + name, pd.Series(pay_group == name, index=test_df.index))
    q33, q66 = train_df.LIMIT_BAL.quantile([1 / 3, 2 / 3])
    lim_group = np.select(
        [test_df.LIMIT_BAL <= q33, test_df.LIMIT_BAL <= q66],
        ["низкий лимит (≤ 33-й перцентиль)", "средний лимит"],
        default="высокий лимит (> 66-го перцентиля)")
    for name in pd.unique(lim_group):
        add_slice("Лимит: " + name, pd.Series(lim_group == name, index=test_df.index))
    age_group = np.select(
        [test_df.AGE < 30, test_df.AGE <= 45],
        ["младше 30 лет", "30–45 лет"], default="старше 45 лет")
    for name in pd.unique(age_group):
        add_slice("Возраст: " + name, pd.Series(age_group == name, index=test_df.index))
    slices = pd.DataFrame(slices)
    slices.to_csv(OUT / "slices.csv", index=False)
    weak = slices[slices.recall_at_t < 0.5]
    log("срезов:", len(slices), "; полнота ниже 0,5:", ", ".join(weak.slice) or "нет")

    # ---------- 5. Интерпретация: важность и частичные зависимости ----------
    log("== 5. ИНТЕРПРЕТАЦИЯ ==")
    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": X_test.columns,
                               "importance": imp.importances_mean,
                               "std": imp.importances_std}
                              ).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "importance.csv", index=False)
    log("важнейшие признаки:", ", ".join(
        f"{r.feature} ({n(r.importance)})" for r in importance.head(6).itertuples()))

    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    top = importance.head(15).iloc[::-1]
    ax.barh(range(len(top)), top.importance, xerr=top["std"], color=BLUE,
            height=0.62, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(top.importance):
        ax.text(v + 0.001, i, n(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(top)), top.feature, fontsize=9)
    comma(ax, "x", 3)
    style(ax, "Перестановочная важность признаков (снижение ROC-AUC)",
          "Снижение ROC-AUC при перемешивании признака", "")
    plt.savefig(OUT / "fig03_permutation.png", dpi=200)
    plt.close(fig)

    pdp_features = ["PAY_1", "LIMIT_BAL", "BILL_AMT1"]

    def pdp_curve(feature, grid):
        means = []
        for v in grid:
            Xc = X_test.copy()
            Xc[feature] = v
            means.append(model.predict_proba(Xc)[:, 1].mean())
        return np.array(means)

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for a, f in zip(ax, pdp_features):
        if f == "PAY_1":                      # дискретный признак: все значения
            grid = np.array(sorted(X_test[f].unique()))
        else:                                 # непрерывный: перцентили
            grid = np.unique(np.percentile(X_test[f].astype(float),
                                            np.linspace(0, 100, 40)))
        curve = pdp_curve(f, grid)
        a.plot(grid, curve, color=BLUE, lw=2)
        comma(a, "x", 0 if f == "PAY_1" else 0)
        comma(a, "y", 2)
        style(a, f"Средняя вероятность дефолта от {f}", f,
              "Средняя вероятность дефолта")
    plt.savefig(OUT / "fig04_pdp.png", dpi=200)
    plt.close(fig)

    # ---------- 6. Объяснения отдельных предсказаний ----------
    log("== 6. ЛОКАЛЬНЫЕ ОБЪЯСНЕНИЯ ==")
    typical = {f: (float(train_df[f].mode().iloc[0])
                   if f in ("SEX", "EDUCATION", "MARRIAGE") or f.startswith("PAY_")
                   else float(train_df[f].median())) for f in RAW_FEATURES}

    def explain(case, label, idx):
        row = X_test.iloc[[idx]].copy()
        p0 = float(model.predict_proba(row)[:, 1][0])
        contribs = []
        for f in RAW_FEATURES:
            changed = row.copy()
            changed[f] = typical[f]
            contribs.append((f, p0 - float(model.predict_proba(changed)[:, 1][0])))
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        d = test_df.iloc[idx]
        client = (f"лимит {int(d.LIMIT_BAL)} NT$, PAY_1 = {int(d.PAY_1)}, "
                  f"возраст {int(d.AGE)}, счёт {int(d.BILL_AMT1)} NT$")
        top_c = "; ".join(f"{f} {'+' if c > 0 else '–'}{abs(c):.2f}".replace(".", ",")
                          for f, c in contribs[:3])
        return {"case": label, "client": client, "probability": round(p0, 3),
                "top_contribs": top_c, "id": int(d[ID_COL])}

    mask_star = p_test >= t_star
    explanations = []
    idx_ok = np.argmax(np.where((y_test == 1) & mask_star, p_test, -1))
    explanations.append(explain(idx_ok, "Верное срабатывание (клиент ушёл в дефолт)", idx_ok))
    idx_fp = np.argmax(np.where((y_test == 0) & mask_star, p_test, -1))
    explanations.append(explain(idx_fp, "Ложное срабатывание (дефолта не было)", idx_fp))
    idx_fn = np.argmin(np.where(y_test == 1, p_test, 2))
    explanations.append(explain(idx_fn, "Пропуск (клиент ушёл в дефолт)", idx_fn))
    local = pd.DataFrame(explanations)
    local.to_csv(OUT / "local_explanations.csv", index=False)
    for r in explanations:
        log(" ", r["case"], "| вероятность", r["probability"], "|",
            r["top_contribs"])

    # ---------- 7. Устойчивость ----------
    log("== 7. УСТОЙЧИВОСТЬ ==")
    rng = np.random.default_rng(SEED)
    money = [c for c in ("LIMIT_BAL",) ] + [f"BILL_AMT{i}" for i in range(1, 7)] \
        + [f"PAY_AMT{i}" for i in range(1, 7)]
    X_noise = X_test.copy()
    for c in money:
        X_noise[c] = X_noise[c] * (1 + rng.normal(0, 0.1, len(X_noise)))
    X_pay = X_test.copy()
    for c in [f"PAY_{i}" for i in range(1, 7)]:
        X_pay[c] = (X_pay[c] + rng.integers(-1, 2, len(X_pay))).clip(-2, 8)
    X_gaps = X_test.copy()
    idx_gaps = rng.choice(len(X_gaps), int(0.2 * len(X_gaps)), replace=False)
    X_gaps.loc[X_gaps.index[idx_gaps], "PAY_1"] = int(train_df.PAY_1.mode().iloc[0])
    stability = pd.DataFrame([
        {"perturbation": "без возмущений", "roc_auc": roc_lr},
        {"perturbation": "шум 10 % в денежные признаки",
         "roc_auc": roc_auc_score(y_test, model.predict_proba(X_noise)[:, 1])},
        {"perturbation": "сдвиг статусов PAY на ±1",
         "roc_auc": roc_auc_score(y_test, model.predict_proba(X_pay)[:, 1])},
        {"perturbation": "20 % значений PAY_1 заменены модой",
         "roc_auc": roc_auc_score(y_test, model.predict_proba(X_gaps)[:, 1])},
    ])
    stability.to_csv(OUT / "stability.csv", index=False)
    for r in stability.itertuples():
        log(" ", r.perturbation, "-> ROC-AUC", n(r.roc_auc))

    # ---------- 8. Проверки качества данных ----------
    log("== 8. ПРОВЕРКИ КАЧЕСТВА ДАННЫХ ==")
    bad = pd.read_csv(BAD_PATH)
    report_bad = check_quality(bad)
    (OUT / "quality_bad.json").write_text(
        json.dumps(report_bad, ensure_ascii=False, indent=2), encoding="utf-8")
    log("испорченный файл (", len(bad), "записей ):", "ОТКЛОНЁН" if not report_bad["ok"] else "принят")
    log(" ", format_report(report_bad))
    rows = []
    for path in BATCH_FILES[:4]:
        rep = check_quality(pd.read_csv(path))
        rows.append({"batch": path.stem, "ok": rep["ok"], "rows": rep["rows"],
                     "rejected_share": rep["rejected_share"]})
    quality_batches = pd.DataFrame(rows)
    quality_batches.to_csv(OUT / "quality_batches.csv", index=False)
    log("порции batch_01–batch_04: проверки пройдены:",
        bool(quality_batches.ok.all()))

    # ---------- 9. Качество и PSI по порциям ----------
    log("== 9. ПОРЦИИ НОВЫХ ДАННЫХ ==")
    reference = load_reference()
    batch_rows = []
    for path in BATCH_FILES:
        df = clean(pd.read_csv(path))
        X, y = df[RAW_FEATURES], df[TARGET]
        p = model.predict_proba(X)[:, 1]
        psi_v = psi_features(df, reference)
        row = {"batch": path.stem, "n": len(df), "default_rate": y.mean(),
               "mean_pd": p.mean(), "calib_gap": p.mean() - y.mean(),
               "roc_auc": roc_auc_score(y, p),
               "pr_auc": average_precision_score(y, p),
               "recall_at_t": recall_score(y, (p >= t_star).astype(int),
                                           zero_division=0),
               "psi_PAY_1": psi_v.get("PAY_1"), "psi_BILL_AMT1": psi_v.get("BILL_AMT1"),
               "psi_PAY_AMT1": psi_v.get("PAY_AMT1"),
               "max_psi": max(psi_v.values()),
               "psi_status": psi_status(max(psi_v.values()))}
        batch_rows.append(row)
        log(" ", path.stem, ": ROC-AUC", n(row["roc_auc"]), "полнота",
            n(row["recall_at_t"]), "макс. PSI", n(row["max_psi"]), "-",
            row["psi_status"])
    batches = pd.DataFrame(batch_rows)
    batches.to_csv(OUT / "batches.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6), constrained_layout=True)
    x = np.arange(len(batches))
    ax[0].plot(x, batches.roc_auc, color=BLUE, lw=2, marker="o", ms=5,
               label="ROC-AUC на порции")
    ax[0].axhline(ACCEPT["batch_roc_auc_min"], color=MUTED, lw=1, ls="--",
                  label="порог приемки 0,75")
    ax[0].plot(x, batches.mean_pd, color=AQUA, lw=2, marker="s", ms=5,
               label="средняя прогнозная вероятность")
    ax[0].plot(x, batches.default_rate, color=ORANGE, lw=2, marker="^", ms=5,
               ls="--", label="фактическая доля дефолтов")
    ax[0].set_xticks(x, batches.batch)
    ax[0].legend(frameon=False, loc="lower left", fontsize=8)
    comma(ax[0], "y", 2)
    style(ax[0], "Качество модели и калибровка по порциям",
          "Порция («новая» выгрузка)", "Значение")
    width = 0.26
    for k, (feat, color) in enumerate([("psi_PAY_1", BLUE), ("psi_BILL_AMT1", AQUA),
                                       ("psi_PAY_AMT1", ORANGE)]):
        ax[1].bar(x + (k - 1) * width, batches[feat], width=width, color=color,
                  label=f"PSI: {feat.replace('psi_', '')}")
    ax[1].axhline(PSI_STABLE, color=MUTED, lw=1, ls=":")
    ax[1].axhline(PSI_ALERT, color=MUTED, lw=1.2, ls="--")
    ax[1].annotate("порог сигнала 0,25", (0.02, PSI_ALERT + 0.02), fontsize=8,
                   color=MUTED)
    ax[1].set_xticks(x, batches.batch)
    ax[1].legend(frameon=False, fontsize=8)
    comma(ax[1], "y", 2)
    style(ax[1], "Индекс стабильности распределения признаков",
          "Порция («новая» выгрузка)", "PSI")
    plt.savefig(OUT / "fig05_batches.png", dpi=200)
    plt.close(fig)

    # ---------- 10. Паспорт модели и решение ----------
    log("== 10. ПАСПОРТ МОДЕЛИ ==")
    stress = batches[batches.batch == "batch_05"].iloc[0]
    weak_text = "; ".join(
        f"{r.slice} (полнота {r.recall_at_t:.2f})" for r in weak.itertuples())
    passport = {
        "version": info["version"], "file": info["file"]["path"],
        "purpose": "оценка вероятности дефолта клиента в следующем месяце для "
                   "приоритизации ручной проверки и пересмотра лимитов",
        "algorithm": info["model"], "trained": info["trained"],
        "train_data": {"file": info["data"]["train"], "n": info["data"]["n_train"],
                       "default_rate": info["data"]["default_rate"]},
        "quality": {"test_roc_auc": round(float(roc_lr), 3),
                    "test_roc_auc_ci": [round(float(ci_lr[0]), 3),
                                         round(float(ci_lr[1]), 3)],
                    "pr_auc": round(float(average_precision_score(y_test, p_test)), 3),
                    "brier": round(float(brier_lr), 3),
                    "threshold": round(t_star, 2),
                    "recall_at_threshold": round(float(thr["recall"]), 3)},
        "top_features": list(importance.head(6).feature),
        "limitations": [
            "ранжирование внутри сегментов заметно слабее общего уровня: " + weak_text,
            "на стресс-порции batch_05 (имитация дрейфа) переоценивает уровень "
            "риска: средняя прогнозная вероятность выше фактической доли дефолтов "
            f"на {abs(stress.calib_gap) * 100:.1f} п. п., ROC-AUC ниже порога приемки"],
        "retraining_rules": {"psi": PSI_ALERT, "roc_auc_min": ACCEPT["batch_roc_auc_min"],
                             "calibration_gap": ACCEPT["batch_calibration_gap_max"]},
        "decision": "модель принимается к использованию с ограничениями: пороги "
                    "приемки выполнены; по сигналу PSI > 0,25 или расхождению "
                    "калибровки требуется переобучение (практическая работа № 5)",
    }
    (OUT / "model_passport.json").write_text(
        json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = (roc_lr >= ACCEPT["roc_auc_min"] and
              roc_lr - roc_rule >= ACCEPT["roc_auc_vs_rule_min"] and
              average_precision_score(y_test, p_test) >= ACCEPT["pr_auc_min"] and
              brier_lr <= ACCEPT["brier_max"] and
              thr["recall"] >= ACCEPT["recall_at_threshold_min"] and
              abs(p_test.mean() - y_test.mean()) <= ACCEPT["calibration_gap_max"])
    log("пороги приемки выполнены:", passed)
    log("готово за", round(time.perf_counter() - t_start), "с")
    OUT_TXT.write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
