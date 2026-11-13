"""ПР3. Обучение и сравнение моделей.

Все сравнения выполняются пятиблочной стратифицированной кросс-валидацией
на обучающей выборке data/train.csv; тестовая выборка не используется.
Каждый эксперимент записывается строкой в reports/experiments.csv
(модель, параметры, состав признаков, метрики по блокам, время обучения, дата).

Этапы: базовые решения -> логистическая регрессия L1/L2 -> модели
классификации -> ансамбли -> польза новых признаков -> подбор
гиперпараметров -> сводная таблица и выбор итоговой модели.

Запуск из корня репозитория (около 3 минут):
    python src/experiments.py
Результаты: reports/experiments.csv, reports/pr3/ (таблицы, графики,
selected_model.json - описание выбранной модели для src/train.py).
"""

import json
import time
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import PAY_COLS, REPORTS_DIR, SEED
from features import CAT_COLS, make_pipeline, prepare
from split_train_test import TRAIN_PATH

OUT = REPORTS_DIR / "pr3"
EXP_PATH = REPORTS_DIR / "experiments.csv"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
CV_NAME = "StratifiedKFold(5, shuffle=True, random_state=42)"
SCORING = {"roc_auc": "roc_auc",                 # основная метрика
           "pr_auc": "average_precision",        # вспомогательная: дисбаланс классов
           "brier": "neg_brier_score",           # вспомогательная: калибровка
           "f1": "f1", "recall": "recall",       # при пороге 0,5 (справочно)
           "precision": make_scorer(precision_score, zero_division=0)}
N_JOBS = -1
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
K_GRID = [5, 15, 25, 50, 100, 200]
DEPTHS = list(range(1, 16)) + [None]
# Правило выбора: наибольший средний ROC-AUC; если модели отличаются от лучшей
# меньше чем на 0,005, берется более простое семейство, а внутри семейства -
# модель с меньшим показателем Брайера (лучше откалиброванные вероятности:
# вероятность дефолта напрямую входит в расчет ожидаемых потерь)
SELECT_TOL = 0.005

# Сложность семейств для правила выбора (меньше - проще и быстрее)
COMPLEXITY = {"LogisticRegression": 1, "GaussianNB": 2, "DecisionTreeClassifier": 3,
              "KNeighborsClassifier": 4, "RandomForestClassifier": 5,
              "HistGradientBoostingClassifier": 6}
ESTIMATORS = {cls.__name__: cls for cls in
              [LogisticRegression, GaussianNB, KNeighborsClassifier,
               DecisionTreeClassifier, RandomForestClassifier,
               HistGradientBoostingClassifier]}

# Цвета серий (проверенная категориальная палитра) и текст
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

ROWS: list[dict] = []


class PayRule(ClassifierMixin, BaseEstimator):
    """Базовое правило: просрочка в последнем месяце (PAY_1 >= 1) - дефолт."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["PAY_1"].to_numpy() >= 1).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (X["PAY_1"].to_numpy() >= 1).astype(int)


def run(stage, name, estimator, X, y, *, est_name="", params=None, kind="",
        features="extended", train_score=False, keep=False, extra=None):
    """Кросс-валидация одного варианта и запись строки в таблицу экспериментов."""
    res = cross_validate(estimator, X, y, cv=CV, scoring=SCORING, n_jobs=N_JOBS,
                         return_train_score=train_score, return_estimator=keep)
    auc = res["test_roc_auc"]
    row = {"exp_id": len(ROWS) + 1, "date": date.today().isoformat(), "stage": stage,
           "model": name, "estimator": est_name, "kind": kind, "features": features,
           "params": json.dumps(params or {}, ensure_ascii=False), "cv": CV_NAME,
           "n_train": len(y),
           "roc_auc_mean": auc.mean(), "roc_auc_std": auc.std(),
           "roc_auc_folds": " ".join(f"{v:.4f}" for v in auc),
           "pr_auc_mean": res["test_pr_auc"].mean(),
           "brier_mean": -res["test_brier"].mean(),
           "f1_mean": res["test_f1"].mean(), "recall_mean": res["test_recall"].mean(),
           "precision_mean": res["test_precision"].mean(),
           "fit_time_s": res["fit_time"].mean()}
    if train_score:
        row["train_roc_auc_mean"] = res["train_roc_auc"].mean()
    row.update(extra or {})
    ROWS.append(row)
    print(f"  [{row['exp_id']:>3}] {name:<48} ROC-AUC {auc.mean():.3f} ± {auc.std():.3f}"
          f"  PR-AUC {row['pr_auc_mean']:.3f}  {row['fit_time_s']:.2f} с")
    return row, res


def model_run(stage, name, cls, params, kind, X, y, features="extended", **kw):
    """Эксперимент с моделью из ESTIMATORS внутри общего пайплайна."""
    pipe = make_pipeline(cls(**params), kind=kind, features=features)
    return run(stage, name, pipe, X, y, est_name=cls.__name__, params=params,
               kind=kind, features=features, **kw)


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
    """Десятичная запятая в подписях осей (оформление отчета на русском языке)."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(fmt)


def lr_params(penalty: str, C: float) -> dict:
    if penalty == "l1":   # liblinear перемешивает данные - нужен random_state
        return {"C": C, "l1_ratio": 1.0, "solver": "liblinear", "random_state": SEED}
    return {"C": C, "l1_ratio": 0.0, "max_iter": 2000}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    plt.rcParams.update({"font.size": 11})
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = prepare(TRAIN_PATH)
    print(f"Обучающая выборка: {len(y)} записей, доля дефолтов {y.mean():.3f}")
    t_start = time.perf_counter()

    # ---------- 1. Базовые решения ----------
    print("1. Базовые решения")
    run("baseline", "Наиболее частый класс", DummyClassifier(strategy="most_frequent"),
        X, y, est_name="DummyClassifier", params={"strategy": "most_frequent"},
        features="base")
    run("baseline", "Правило: просрочка в сентябре (PAY_1 >= 1)", PayRule(), X, y,
        est_name="PayRule", params={"rule": "PAY_1 >= 1"}, features="base")

    # ---------- 2. Логистическая регрессия L1 / L2 ----------
    print("2. Логистическая регрессия")
    reg = []
    for penalty in ["l2", "l1"]:
        for C in C_GRID:
            params = lr_params(penalty, C)
            row, res = model_run("linear", f"Логистическая регрессия {penalty.upper()} (C = {C:g})",
                                 LogisticRegression, params, "linear", X, y, keep=True)
            nonzero = np.mean([(est[-1].coef_ != 0).sum() for est in res["estimator"]])
            n_coef = res["estimator"][0][-1].coef_.size
            row["n_nonzero"] = nonzero
            row["n_coef"] = n_coef
            reg.append({"penalty": penalty.upper(), "C": C, "roc_auc": row["roc_auc_mean"],
                        "roc_auc_std": row["roc_auc_std"], "n_nonzero": nonzero,
                        "n_coef": n_coef})
    reg = pd.DataFrame(reg)
    reg.to_csv(OUT / "regularization.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for pen, color, marker in [("L2", BLUE, "o"), ("L1", ORANGE, "s")]:
        part = reg[reg.penalty == pen]
        ax[0].plot(part.C, part.roc_auc, color=color, marker=marker, lw=2, ms=6,
                   label=f"Регуляризация {pen}")
        ax[1].plot(part.C, part.n_nonzero, color=color, marker=marker, lw=2, ms=6,
                   label=f"Регуляризация {pen}")
    for a in ax:
        a.set_xscale("log")
        a.legend(frameon=False)
    comma(ax[0], "y", 2)
    style(ax[0], "Качество модели", "Параметр C (обратная сила регуляризации)",
          "ROC-AUC на кросс-валидации")
    style(ax[1], "Число ненулевых коэффициентов",
          "Параметр C (обратная сила регуляризации)", "Число коэффициентов, шт.")
    plt.tight_layout()
    plt.savefig(OUT / "fig01_regularization.png", dpi=150)
    plt.close()

    # лучший C для каждого вида регуляризации
    best_c = {pen: float(reg[reg.penalty == pen]
                         .sort_values("roc_auc", ascending=False).C.iloc[0])
              for pen in ["L2", "L1"]}
    print(f"  лучший C: L2 = {best_c['L2']:g}, L1 = {best_c['L1']:g}")

    # анализ коэффициентов L2-модели с лучшим C (обучение на всей обучающей выборке)
    lr = make_pipeline(LogisticRegression(**lr_params("l2", best_c["L2"])), "linear").fit(X, y)
    names = lr.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": lr[-1].coef_[0]})
    # для индикаторов категорий - число записей и доля дефолтов в категории
    n_rec, rate = [], []
    for f in names:
        col, _, val = f.rpartition("_")
        if col in CAT_COLS + PAY_COLS and val.lstrip("-").isdigit():
            mask = X[col] == int(val)
            n_rec.append(int(mask.sum()))
            rate.append(y[mask].mean())
        else:
            n_rec.append(None)
            rate.append(None)
    coefs["n_records"] = n_rec
    coefs["default_rate"] = rate
    coefs["abs"] = coefs.coef.abs()
    coefs.sort_values("abs", ascending=False).drop(columns="abs") \
        .to_csv(OUT / "logreg_coefs.csv", index=False)
    # какие признаки остаются при сильной L1-регуляризации
    l1_rows = []
    for C in [0.003, 0.01, 0.03]:
        m = make_pipeline(LogisticRegression(**lr_params("l1", C)), "linear").fit(X, y)
        kept = [n for n, c in zip(names, m[-1].coef_[0]) if c != 0]
        l1_rows.append({"C": C, "n_nonzero": len(kept), "features": ", ".join(kept)})
    pd.DataFrame(l1_rows).to_csv(OUT / "l1_selected.csv", index=False)

    # ---------- 3. Модели классификации ----------
    print("3. Модели классификации")
    for k in K_GRID:
        model_run("classifiers", f"k ближайших соседей (k = {k})", KNeighborsClassifier,
                  {"n_neighbors": k}, "linear", X, y)
    depth_rows = []
    for d in DEPTHS:
        params = {"max_depth": d, "random_state": SEED}
        label = "без ограничения" if d is None else str(d)
        row, _ = model_run("classifiers", f"Дерево решений (глубина {label})",
                           DecisionTreeClassifier, params, "tree", X, y, train_score=True)
        depth_rows.append({"depth": label, "train_roc_auc": row["train_roc_auc_mean"],
                           "cv_roc_auc": row["roc_auc_mean"], "cv_std": row["roc_auc_std"]})
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(OUT / "tree_depth.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    finite = depth[depth.depth != "без ограничения"]
    xs = finite.depth.astype(int)
    ax.plot(xs, finite.train_roc_auc, color=BLUE, marker="o", lw=2, ms=5,
            label="Обучающие блоки")
    ax.plot(xs, finite.cv_roc_auc, color=ORANGE, marker="s", lw=2, ms=5,
            label="Проверочные блоки")
    best_d = int(finite.sort_values("cv_roc_auc", ascending=False).depth.iloc[0])
    ax.axvline(best_d, color=MUTED, lw=1, ls="--")
    ax.annotate(f"лучшая глубина {best_d}", (best_d, finite.cv_roc_auc.min()),
                xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_xticks(xs)
    ax.legend(frameon=False)
    comma(ax, "y", 2)
    style(ax, "Дерево решений: качество в зависимости от глубины",
          "Максимальная глубина дерева, уровней", "ROC-AUC (среднее по 5 блокам)")
    plt.tight_layout()
    plt.savefig(OUT / "fig02_tree_depth.png", dpi=150)
    plt.close()

    model_run("classifiers", "Наивный байесовский классификатор", GaussianNB, {},
              "linear", X, y)

    # ---------- 4. Ансамбли ----------
    print("4. Ансамбли")
    rf_default = {"n_estimators": 300, "random_state": SEED}
    hgb_default = {"random_state": SEED}
    model_run("ensembles", "Случайный лес (300 деревьев)", RandomForestClassifier,
              rf_default, "tree", X, y)
    model_run("ensembles", "Градиентный бустинг (по умолчанию)",
              HistGradientBoostingClassifier, hgb_default, "tree", X, y)

    # ---------- 5. Польза новых признаков ----------
    print("5. Польза новых признаков: исходные 23 признака против 38")
    gain = []
    checks = [("Логистическая регрессия L2", LogisticRegression,
               lr_params("l2", best_c["L2"]), "linear"),
              ("Дерево решений", DecisionTreeClassifier,
               {"max_depth": best_d, "random_state": SEED}, "tree"),
              ("Случайный лес", RandomForestClassifier, rf_default, "tree"),
              ("Градиентный бустинг", HistGradientBoostingClassifier, hgb_default, "tree")]
    for name, cls, params, kind in checks:
        ext = next(r for r in ROWS if r["estimator"] == cls.__name__
                   and r["params"] == json.dumps(params, ensure_ascii=False)
                   and r["features"] == "extended")
        base, _ = model_run("features", f"{name} (исходные признаки)", cls, params,
                            kind, X, y, features="base")
        gain.append({"model": name, "base_roc_auc": base["roc_auc_mean"],
                     "extended_roc_auc": ext["roc_auc_mean"],
                     "delta": ext["roc_auc_mean"] - base["roc_auc_mean"],
                     "base_pr_auc": base["pr_auc_mean"], "extended_pr_auc": ext["pr_auc_mean"]})
    pd.DataFrame(gain).to_csv(OUT / "feature_gain.csv", index=False)

    # ---------- 6. Подбор гиперпараметров ----------
    print("6. Подбор гиперпараметров")
    # две лучшие модели по итогам этапов 2-4: логистическая регрессия и бустинг
    c_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    weights = [None, "balanced"]
    grids = {
        "LogisticRegression": (
            "Логистическая регрессия (после подбора)", "linear",
            [{"model__l1_ratio": [0.0], "model__solver": ["lbfgs"],
              "model__C": c_values, "model__class_weight": weights},
             {"model__l1_ratio": [1.0], "model__solver": ["liblinear"],
              "model__C": c_values, "model__class_weight": weights},
             {"model__l1_ratio": [0.5], "model__solver": ["saga"],
              "model__C": c_values, "model__class_weight": weights}],
            {"max_iter": 5000, "random_state": SEED}),
        "HistGradientBoostingClassifier": (
            "Градиентный бустинг (после подбора)", "tree",
            {"model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [3, 5, None],
             "model__max_iter": [200, 400],
             "model__min_samples_leaf": [20, 100]},
            {"early_stopping": False, "random_state": SEED}),
    }
    for est_name, (label, kind, grid, fixed) in grids.items():
        cls = ESTIMATORS[est_name]
        search = GridSearchCV(make_pipeline(cls(**fixed), kind), grid, cv=CV,
                              scoring={"roc_auc": "roc_auc", "brier": "neg_brier_score"},
                              refit="roc_auc", n_jobs=N_JOBS)
        t0 = time.perf_counter()
        search.fit(X, y)
        n_comb = len(search.cv_results_["params"])
        cvres = pd.DataFrame(search.cv_results_)
        cvres["mean_test_brier"] = -cvres["mean_test_brier"]
        keep_cols = [c for c in cvres.columns if c.startswith("param_")] +             ["mean_test_roc_auc", "std_test_roc_auc", "mean_test_brier",
             "mean_fit_time", "rank_test_roc_auc"]
        cvres[keep_cols].sort_values("rank_test_roc_auc")             .to_csv(OUT / f"tuning_{est_name}.csv", index=False)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        params = {**fixed, **best}
        print(f"  {est_name}: {n_comb} сочетаний, {time.perf_counter() - t0:.0f} с, "
              f"лучшие {best}, ROC-AUC {search.best_score_:.4f}")
        model_run("tuning", label, cls, params, kind, X, y,
                  extra={"n_combinations": n_comb})

    # калибровка: средняя прогнозная вероятность дефолта с весами классов и без них
    calib = []
    for cw in [None, "balanced"]:
        params = {**lr_params("l1", best_c["L1"]), "class_weight": cw}
        proba = cross_val_predict(make_pipeline(LogisticRegression(**params), "linear"),
                                  X, y, cv=CV, method="predict_proba",
                                  n_jobs=N_JOBS)[:, 1]
        calib.append({"model": f"Логистическая регрессия L1 (C = {best_c['L1']:g})",
                      "class_weight": cw or "нет", "mean_pd": proba.mean(),
                      "default_rate": y.mean(), "brier": ((proba - y) ** 2).mean()})
        print(f"  веса классов {cw or 'нет'}: средняя вероятность дефолта "
              f"{proba.mean():.3f} при доле дефолтов {y.mean():.3f}")
    pd.DataFrame(calib).to_csv(OUT / "class_weight_calibration.csv", index=False)

    # ---------- 7. Сводная таблица и выбор модели ----------
    print("7. Сводная таблица и выбор итоговой модели")
    exp = pd.DataFrame(ROWS)
    exp.to_csv(EXP_PATH, index=False, float_format="%.4f")

    def best_of(stage, est):
        part = exp[(exp.stage == stage) & (exp.estimator == est)
                   & (exp.features == "extended")]
        return part.loc[part.roc_auc_mean.idxmax()]

    summary = pd.concat([
        exp[exp.stage == "baseline"],
        pd.DataFrame([exp[(exp.stage == "linear") & exp.model.str.contains(pen)]
                      .sort_values("roc_auc_mean").iloc[-1] for pen in ["L2", "L1"]]),
        pd.DataFrame([best_of("classifiers", "KNeighborsClassifier")]),
        pd.DataFrame([best_of("classifiers", "DecisionTreeClassifier")]),
        exp[exp.model == "Дерево решений (глубина без ограничения)"],
        exp[(exp.stage == "classifiers") & (exp.estimator == "GaussianNB")],
        exp[exp.stage == "ensembles"],
        exp[exp.stage == "tuning"],
    ])
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    cand = summary[summary.stage != "baseline"].copy()
    top = cand.roc_auc_mean.max()
    near = cand[cand.roc_auc_mean >= top - SELECT_TOL].copy()
    near["complexity"] = near.estimator.map(COMPLEXITY)
    chosen = near.sort_values(["complexity", "brier_mean"]).iloc[0]
    print(f"  наибольший ROC-AUC {top:.4f}; в пределах {SELECT_TOL}: "
          f"{', '.join(near.model)}")
    print(f"  выбрана: {chosen.model} (эксперимент {chosen.exp_id})")
    selected = {
        "exp_id": int(chosen.exp_id), "model": chosen.model,
        "estimator": chosen.estimator, "kind": chosen.kind, "features": chosen.features,
        "params": json.loads(chosen.params),
        "cv": {"scheme": CV_NAME, "roc_auc_mean": round(chosen.roc_auc_mean, 4),
               "roc_auc_std": round(chosen.roc_auc_std, 4),
               "pr_auc_mean": round(chosen.pr_auc_mean, 4),
               "brier_mean": round(chosen.brier_mean, 4)},
        "rule": f"наибольший средний ROC-AUC; при разнице меньше {SELECT_TOL} "
                "выбирается более простое семейство, внутри семейства - "
                "модель с меньшим показателем Брайера",
        "candidates_within_tol": list(near.model),
    }
    (OUT / "selected_model.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    # график сравнения
    plot = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    colors = [ORANGE if e == chosen.exp_id else (MUTED if s == "baseline" else BLUE)
              for e, s in zip(plot.exp_id, plot.stage)]
    ax.barh(range(len(plot)), plot.roc_auc_mean, xerr=plot.roc_auc_std, color=colors,
            height=0.6, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for i, (v, sd) in enumerate(zip(plot.roc_auc_mean, plot.roc_auc_std)):
        ax.text(v + sd + 0.006, i, f"{v:.3f}".replace(".", ","), va="center",
                fontsize=10, color=INK)
    labels = [m.replace(">=", "≥").replace("0.", "0,") for m in plot.model]
    ax.set_yticks(range(len(plot)), labels, fontsize=10)
    ax.set_xlim(0.45, 0.85)
    comma(ax, "x", 2)
    style(ax, "ROC-AUC: среднее ± ст. откл. по 5 блокам",
          "ROC-AUC на кросс-валидации", "")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label="Базовое решение"),
                       Patch(color=BLUE, label="Модель"),
                       Patch(color=ORANGE, label="Выбранная модель")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=3)
    plt.tight_layout()
    plt.savefig(OUT / "fig03_comparison.png", dpi=150)
    plt.close()

    print(f"Готово за {time.perf_counter() - t_start:.0f} с. "
          f"Экспериментов: {len(exp)} -> {EXP_PATH.relative_to(REPORTS_DIR.parent).as_posix()}")


if __name__ == "__main__":
    main()
