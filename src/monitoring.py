"""ПР4/ПР5. Показатели мониторинга: индекс стабильности распределения (PSI)
и правила контроля внедренной модели.

PSI рассчитывается по квантильным интервалам эталонной выборки (обучающая
выборка data/train.csv): значения признака разбиваются на интервалы по
десяти децилям эталона, и сравниваются доли объектов порции в этих
интервалах. Для дискретных признаков (не более 12 значений) сравниваются
доли значений. Эталонные распределения сохраняются в файл
models/reference_stats.json, поэтому при обработке порций обучающая
выборка не нужна.

Интерпретация (пособие, п. 4.2.4): PSI < 0,1 - распределение стабильно;
0,1–0,25 - умеренное изменение; > 0,25 - существенное изменение, сигнал к
переобучению.
"""

import json

import numpy as np
import pandas as pd

from config import MODELS_DIR
from features import clean
from split_train_test import TRAIN_PATH

REFERENCE_PATH = MODELS_DIR / "reference_stats.json"
EPS = 1e-6
# Ключевые признаки для контроля дрейфа: главный предиктор, сумма счёта и
# платёж за последний месяц (выбор обоснован перестановочной важностью, ПР4)
PSI_FEATURES = ["PAY_1", "BILL_AMT1", "PAY_AMT1"]

# Пороги мониторинга (из ПР4: пороги приемки и план проверки на новых данных)
PSI_STABLE, PSI_MODERATE, PSI_ALERT = 0.1, 0.25, 0.25
ROC_AUC_MIN = 0.75          # минимум качества на размеченной порции
CALIB_GAP_MAX = 0.05        # допустимое расхождение прогнозной и фактической доли
RISK_JUMP_MAX = 0.05        # рост доли зоны риска между порциями, п. п.


def _shares(values, kind, edges=None):
    """Доли значений по интервалам: категории или квантильные интервалы."""
    if kind == "cat":
        cats = sorted(edges)
        counts = [int((values == c).sum()) for c in cats]
        total = sum(counts) or 1
        return np.array([c / total for c in counts])
    counts, _ = np.histogram(values, bins=edges)
    return counts / (counts.sum() or 1)


def make_reference() -> dict:
    """Эталонные распределения ключевых признаков по обучающей выборке."""
    ref = {}
    train = clean(pd.read_csv(TRAIN_PATH))
    for f in PSI_FEATURES:
        v = train[f].to_numpy()
        if v.size and np.unique(v).size <= 12:
            ref[f] = {"kind": "cat", "edges": sorted(np.unique(v).tolist()),
                      "shares": _shares(v, "cat", np.unique(v)).round(6).tolist()}
        else:
            edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
            if len(edges) < 2:                      # константный признак
                ref[f] = {"kind": "cat", "edges": [float(v[0])],
                          "shares": [1.0]}
                continue
            ref[f] = {"kind": "bin", "edges": edges.tolist(),
                      "shares": _shares(v, "bin", edges).round(6).tolist()}
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(ref, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return ref


def load_reference() -> dict:
    if REFERENCE_PATH.exists():
        return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    return make_reference()


def psi(expected_shares, actual_shares) -> float:
    """Индекс стабильности распределения: sum (a - e) * ln(a / e)."""
    e = np.clip(np.asarray(expected_shares, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_shares, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_features(df: pd.DataFrame, reference: dict | None = None) -> dict:
    """PSI ключевых признаков порции относительно эталона (обучающей выборки)."""
    reference = reference or load_reference()
    result = {}
    for f, spec in reference.items():
        if f not in df.columns:
            continue
        result[f] = round(psi(spec["shares"], _shares(df[f].to_numpy(),
                                                      spec["kind"], spec["edges"])), 4)
    return result


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "стабильно"
    if value < PSI_MODERATE:
        return "умеренное изменение"
    return "существенное изменение"


# ---- Правила мониторинга (ПР5): уровни, пороги, реакции ----
def evaluate_alerts(cur: dict, prev: dict | None) -> list[dict]:
    """Предупреждения по правилам мониторинга для очередной порции.

    cur, prev - словари показателей порции (ключи как в metrics_history):
    risk_share, rejected_share, psi_*, roc_auc, calib_gap.
    Возвращает список {"level": "warning"|"info", "text": ...}.
    """
    alerts = []

    def get(d, k, default=None):
        return d.get(k, default) if d else default

    # предметная область: резкий рост доли зоны риска
    if prev is not None:
        jump = get(cur, "risk_share", 0) - get(prev, "risk_share", 0)
        if jump > RISK_JUMP_MAX:
            alerts.append({"level": "warning",
                           "text": f"доля зоны риска выросла на {jump * 100:.1f} п. п."})
    # данные: дрейф признаков
    for k, v in cur.items():
        if k.startswith("psi_") and v is not None:
            if v >= PSI_ALERT:
                alerts.append({"level": "warning", "text":
                               f"существенное изменение признака {k[4:]}: "
                               f"PSI = {v:.2f} - сигнал к переобучению"})
            elif v >= PSI_MODERATE:
                alerts.append({"level": "info", "text":
                               f"умеренное изменение признака {k[4:]}: PSI = {v:.2f}"})
    # данные: качество поставки
    if get(cur, "rejected_share", 0) > 0.05:
        alerts.append({"level": "warning", "text":
                       "доля отклонённых записей выше 5 % - проверить источник данных"})
    # модель: качество на размеченной порции
    if get(cur, "roc_auc") is not None:
        if cur["roc_auc"] < ROC_AUC_MIN:
            alerts.append({"level": "warning", "text":
                           f"ROC-AUC на порции {cur['roc_auc']:.3f} ниже порога "
                           f"{ROC_AUC_MIN:.2f} - сигнал к переобучению"})
        gap = cur.get("calib_gap")
        if gap is not None and abs(gap) > CALIB_GAP_MAX:
            alerts.append({"level": "warning", "text":
                           f"расхождение прогнозной и фактической доли дефолтов "
                           f"{gap * 100:+.1f} п. п. - учитывается в правиле замены"})
    return alerts
