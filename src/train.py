"""ПР3. Обучение итоговой модели на всей обучающей выборке и сохранение.

Модель и ее параметры берутся из reports/pr3/selected_model.json, который
формирует src/experiments.py по правилу выбора. Сохраняются:
  models/model_v1.joblib - объект Pipeline (признаки, предобработка, модель);
  models/model_v1.json   - описание, достаточное для воспроизведения.

Запуск из корня репозитория:
    python src/train.py
Загрузка модели (модуль src/features.py должен быть доступен для импорта):
    import sys; sys.path.insert(0, "src")
    import joblib; model = joblib.load("models/model_v1.joblib")
    proba = model.predict_proba(X)[:, 1]
"""

import hashlib
import json
import platform
import subprocess
import time
from datetime import date

import joblib
import numpy as np
import pandas as pd
import sklearn

from config import MODELS_DIR, REPORTS_DIR, ROOT, SEED
from experiments import ESTIMATORS
from features import NEW_FEATURES, RAW_FEATURES, feature_columns, make_pipeline, prepare
from split_train_test import TEST_PATH, TRAIN_PATH

VERSION = "v1"
SELECTED_PATH = REPORTS_DIR / "pr3" / "selected_model.json"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    sel = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    X, y = prepare(TRAIN_PATH)
    model = make_pipeline(ESTIMATORS[sel["estimator"]](**sel["params"]),
                          kind=sel["kind"], features=sel["features"])
    t0 = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - t0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{VERSION}.joblib"
    joblib.dump(model, model_path)

    # контроль: модель, загруженная из файла, дает те же вероятности
    proba = model.predict_proba(X)[:, 1]
    same = np.array_equal(joblib.load(model_path).predict_proba(X)[:, 1], proba)

    info = {
        "version": VERSION,
        "trained": date.today().isoformat(),
        "task": "вероятность дефолта клиента в следующем месяце (dpnm = 1)",
        "algorithm": sel["estimator"],
        "model": sel["model"],
        "params": sel["params"],
        "preprocessing": sel["kind"],
        "features": {"set": sel["features"], "n": len(feature_columns(sel["features"])),
                     "raw": RAW_FEATURES,
                     "new": NEW_FEATURES if sel["features"] == "extended" else []},
        "data": {"train": TRAIN_PATH.relative_to(ROOT).as_posix(),
                 "train_md5": md5(TRAIN_PATH), "n_train": int(len(y)),
                 "default_rate": round(float(y.mean()), 4),
                 "test": TEST_PATH.relative_to(ROOT).as_posix(),
                 "test_md5": md5(TEST_PATH),
                 "note": "тестовая выборка при обучении и выборе модели не использовалась"},
        "selection": {"experiment_id": sel["exp_id"], "rule": sel["rule"],
                      "candidates_within_tol": sel["candidates_within_tol"]},
        "cv_metrics": sel["cv"],
        "seed": SEED,
        "fit_time_s": round(fit_time, 2),
        "environment": {"python": platform.python_version(),
                        "scikit-learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "joblib": joblib.__version__},
        "code": {"git_commit": git_commit(),
                 "train": "python src/split_train_test.py && python src/experiments.py "
                          "&& python src/train.py"},
        "file": {"path": model_path.relative_to(ROOT).as_posix(), "md5": md5(model_path)},
    }
    info_path = MODELS_DIR / f"model_{VERSION}.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Итоговая модель: {sel['model']}")
    print(f"Параметры: {sel['params']}")
    print(f"Обучена на {len(y)} записях за {fit_time:.1f} с; "
          f"ROC-AUC на кросс-валидации {sel['cv']['roc_auc_mean']:.3f}")
    print(f"Модель из файла дает те же вероятности: {same}")
    print(f"Сохранено: {model_path.relative_to(ROOT).as_posix()}, "
          f"{info_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
