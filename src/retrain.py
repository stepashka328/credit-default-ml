"""ПР5. Переобучение модели и управление версиями.

Правило переобучения (пороги из ПР4): модель обучается заново, если PSI
ключевого признака превысил 0,25 или ROC-AUC на размеченной порции ниже
0,75. Претендент обучается на накопленных размеченных данных: обучающей
выборке, принятых порциях и 70 % последней порции; порция batch_04 в
обучение не входит: её записи вошли в стресс-порцию batch_05 с
изменёнными признаками, и одни и те же клиенты не должны встретиться и в
обучении претендента, и в его проверке. Оставшиеся 30 % последней порции
образуют контрольную выборку, на которой сравниваются действующая модель и
претендент.

Правило замены: претендент заменяет действующую модель, если его ROC-AUC
не хуже более чем на 0,01, а расхождение прогнозной и фактической доли
дефолтов меньше по модулю. Откат выполняется восстановлением указателя
models/current.json на предыдущую версию.

Запуск из корня репозитория:
    python src/retrain.py               # обучение претендента и замена
    python src/retrain.py --rollback v1 # откат указателя на версию v1
Результат: models/model_v2.joblib, model_v2.json, models/current.json,
           reports/pr5/retrain_comparison.csv
"""

import json
import sys
import time
from datetime import date

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import (CURRENT_MODEL, INCOMING_DIR, MODELS_DIR, REPORTS_DIR,
                    SEED, TARGET)
from features import RAW_FEATURES, clean, make_pipeline
from split_train_test import TRAIN_PATH

OUT = REPORTS_DIR / "pr5"
HOLDOUT_SHARE = 0.3
TOLERANCE = 0.01          # допустимое снижение ROC-AUC претендента
# гиперпараметры сохранены из ПР3: подбор там показал их оптимальность,
# а повторный подбор на новых данных - задание повышенной сложности
PARAMS = {"C": 0.3, "l1_ratio": 1.0, "solver": "liblinear", "random_state": SEED}


def candidate_data() -> tuple:
    """Накопленные данные претендента и контрольная выборка (30 % batch_05)."""
    train = clean(pd.read_csv(TRAIN_PATH))
    batches = [clean(pd.read_csv(INCOMING_DIR / f"batch_{i:02d}.csv"))
               for i in (1, 2, 3)]
    stress = clean(pd.read_csv(INCOMING_DIR / "batch_05.csv"))
    part, hold = train_test_split(stress, test_size=HOLDOUT_SHARE,
                                   stratify=stress[TARGET], random_state=SEED)
    data = pd.concat([train] + batches + [part], ignore_index=True)
    return data, hold


def quality(model, X, y, threshold) -> dict:
    proba = model.predict_proba(X)[:, 1]
    return {"roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba),
            "recall_at_t": recall_score(y, (proba >= threshold).astype(int),
                                        zero_division=0),
            "mean_pd": float(proba.mean()), "actual_rate": float(y.mean()),
            "calib_gap": float(proba.mean() - y.mean())}


def rollback(version: str) -> None:
    model_file = MODELS_DIR / f"model_{version}.joblib"
    if not model_file.exists():
        sys.exit(f"версия {version} не найдена: {model_file}")
    cur = json.loads(CURRENT_MODEL.read_text(encoding="utf-8"))
    old = cur["version"]
    cur["version"] = version
    CURRENT_MODEL.write_text(json.dumps(cur, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"Указатель current.json восстановлен на версию {version} (действовала "
          f"{old}); откат выполнен без переобучения")


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--rollback":
        rollback(sys.argv[2])
        return

    OUT.mkdir(parents=True, exist_ok=True)
    current = json.loads(CURRENT_MODEL.read_text(encoding="utf-8"))
    threshold = float(current["threshold"])
    old_version = current["version"]
    v1 = joblib.load(MODELS_DIR / f"model_{old_version}.joblib")

    data, hold = candidate_data()
    X_train, y_train = data[RAW_FEATURES], data[TARGET]
    X_hold, y_hold = hold[RAW_FEATURES], hold[TARGET]

    t0 = time.perf_counter()
    v2 = make_pipeline(LogisticRegression(**PARAMS), kind="linear")
    v2.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    q1 = quality(v1, X_hold, y_hold, threshold)
    q2 = quality(v2, X_hold, y_hold, threshold)
    replace = (q2["roc_auc"] >= q1["roc_auc"] - TOLERANCE
               and abs(q2["calib_gap"]) < abs(q1["calib_gap"]))

    comparison = pd.DataFrame([
        {"model": f"действующая ({old_version})", **{k: round(v, 4)
                                                      for k, v in q1.items()}},
        {"model": "претендент (v2)", **{k: round(v, 4) for k, v in q2.items()}}])
    comparison.to_csv(OUT / "retrain_comparison.csv", index=False)
    print(f"Обучение претендента на {len(y_train)} записях за {fit_time:.1f} с; "
          f"контрольная выборка: {len(y_hold)} записей (30 % batch_05)")
    print(comparison.to_string(index=False))
    diff = q1["roc_auc"] - q2["roc_auc"]
    print(f"разность ROC-AUC (действующая − претендент): {diff:+.4f} "
          f"(претендент заменяет при разности не более {TOLERANCE}); "
          f"расхождение калибровки: {q2['calib_gap'] * 100:+.1f} п. п. против "
          f"{q1['calib_gap'] * 100:+.1f} п. п. у действующей")

    if not replace:
        print("Решение: претендент НЕ заменяет действующую модель "
              "(правило замены не выполнено)")
        return

    version = "v2"
    model_path = MODELS_DIR / f"model_{version}.joblib"
    joblib.dump(v2, model_path)
    info = {
        "version": version, "trained": date.today().isoformat(),
        "task": "вероятность дефолта клиента в следующем месяце (dpnm = 1)",
        "algorithm": "LogisticRegression",
        "model": "Логистическая регрессия L1 (C = 0.3)",
        "params": PARAMS, "preprocessing": "linear",
        "data": {"train": ["data/train.csv", "data/incoming/batch_01.csv",
                           "data/incoming/batch_02.csv", "data/incoming/batch_03.csv",
                           "70 % data/incoming/batch_05.csv"],
                 "n_train": int(len(y_train)),
                 "note": "batch_04 исключен: его записи входят в стресс-порцию "
                         "batch_05; контроль - 30 % batch_05",
                 "holdout": {"n": int(len(y_hold)),
                             "roc_auc_v1": round(q1["roc_auc"], 4),
                             "roc_auc_v2": round(q2["roc_auc"], 4)}},
        "selection": {"rule": "претендент заменяет действующую модель, если "
                              "ROC-AUC не хуже более чем на 0,01, а расхождение "
                              "прогнозной и фактической доли дефолтов меньше",
                      "replaced": True},
        "seed": SEED, "fit_time_s": round(fit_time, 2),
        "file": {"path": model_path.as_posix()},
        "previous": old_version,
    }
    (MODELS_DIR / f"model_{version}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    current.update({"version": version, "model_file": f"model_{version}.joblib",
                    "previous": old_version, "created": date.today().isoformat()})
    CURRENT_MODEL.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(f"Решение: претендент ЗАМЕНЯЕТ действующую модель -> {version}; "
          f"откат: python src/retrain.py --rollback {old_version}")


if __name__ == "__main__":
    main()
