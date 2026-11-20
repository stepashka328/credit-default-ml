"""ПР5. Сценарий обновления: обработка новой порции одной командой.

Выполняет для порции всю цепочку (пособие, п. 5.2.2): загрузка ->
проверка качества (quality.py, работа № 4) -> подготовка признаков и
предсказание (пайплайн из работы № 3) -> сохранение предсказаний ->
расчет ключевых показателей (практическая работа № 2) -> запись в журнал.

Порция, не прошедшая блокирующие проверки, не обрабатывается: причина
записывается в журнал, данные в хранилище не попадают. Идемпотентность
обеспечивают первичные ключи хранилища (INSERT OR REPLACE): повторная
обработка той же порции перезаписывает результат, не создавая дубликатов.

Запуск из корня репозитория:
    python src/process_batch.py data/incoming/batch_01.csv
Результат: записи в data/credit_system.db и JSON-сводка в стандартном выводе.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from config import CURRENT_MODEL, DB_PATH, ID_COL, MODELS_DIR, TARGET
from features import RAW_FEATURES, clean
from monitoring import evaluate_alerts, load_reference, psi_features
from quality import check_quality
from storage import (connect, init_db, latest_metrics, log_run,
                     previous_metrics, save_metrics, save_predictions, save_raw)


def load_current() -> dict:
    """Указатель на действующую версию модели (models/current.json)."""
    return json.loads(CURRENT_MODEL.read_text(encoding="utf-8"))


def load_model() -> object:
    cur = load_current()
    return joblib.load(MODELS_DIR / f"model_{cur['version']}.joblib")


def compute_metrics(df: pd.DataFrame, proba, threshold: float) -> dict:
    """Ключевые показатели порции (таблица 6 ПР2) и качество модели."""
    metrics = {
        "n_clients": len(df),
        "risk_share": float((proba >= threshold).mean()),
        "expected_loss_kntd": float((df.LIMIT_BAL * proba).sum() / 1000),
        "mean_pd": float(proba.mean()),
    }
    if TARGET in df.columns:                      # фактические значения известны
        y = df[TARGET]
        metrics["actual_rate"] = float(y.mean())
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
        metrics["pr_auc"] = float(average_precision_score(y, proba))
        metrics["recall_at_t"] = float(recall_score(y, (proba >= threshold).astype(int),
                                                    zero_division=0))
        metrics["calib_gap"] = float(proba.mean() - y.mean())
    return metrics


def process_batch(path: str, db_path=DB_PATH) -> dict:
    t0 = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    batch = Path(path).stem

    con = connect(db_path)
    init_db(con)
    raw = pd.read_csv(path)
    report = check_quality(raw)

    if not report["ok"]:
        log_run(con, batch, started_at, "rejected", len(raw),
                time.perf_counter() - t0, "",
                "; ".join(report["failed"]))
        con.close()
        return {"batch": batch, "status": "rejected", "rows": len(raw),
                "failed": report["failed"]}

    current = load_current()
    model = load_model()
    df = clean(raw)
    proba = model.predict_proba(df[RAW_FEATURES])[:, 1]
    threshold = float(current["threshold"])

    save_raw(con, df, batch)
    save_predictions(con, batch, df[ID_COL], proba, proba >= threshold,
                     current["version"])

    metrics = compute_metrics(df, proba, threshold)
    metrics["rejected_share"] = report["rejected_share"]
    metrics.update({f"psi_{k}": v for k, v in psi_features(df).items()})
    save_metrics(con, batch, metrics, started_at)

    alerts = evaluate_alerts(metrics, previous_metrics(con))
    message = "; ".join(a["text"] for a in alerts if a["level"] == "warning")
    log_run(con, batch, started_at, "ok", len(df),
            time.perf_counter() - t0, current["version"], message)
    con.close()
    return {"batch": batch, "status": "ok", "rows": int(len(df)),
            "model_version": current["version"], "duration_s":
            round(time.perf_counter() - t0, 2), **{k: round(v, 4)
            for k, v in metrics.items()}, "alerts": alerts}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("использование: python src/process_batch.py <файл порции>")
    print(json.dumps(process_batch(sys.argv[1]), ensure_ascii=False, indent=1))
