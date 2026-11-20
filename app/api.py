"""ПР5. Программный интерфейс системы (REST, FastAPI).

Методы:
  POST /predict - вероятность дефолта для одного клиента;
  GET  /metrics - ключевые показатели последней обработанной порции;
  GET  /health  - состояние сервиса и версия действующей модели.

Формат запроса описан классом Client: недопустимые значения (отрицательный
лимит, возраст вне диапазона, неизвестный статус PAY) отклоняются с кодом
422 до обращения к модели. API использует тот же файл модели, что и сценарий
обновления (models/current.json), поэтому предсказания по одному клиенту и
по порции согласованы.

Запуск из корня репозитория:
    uvicorn app.api:app --port 8000
Документация: http://localhost:8000/docs
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DB_PATH, MODELS_DIR  # noqa: E402
from process_batch import load_current  # noqa: E402
from storage import connect, latest_metrics, metrics_table  # noqa: E402

app = FastAPI(title="Система анализа риска дефолта",
              description="Вероятность дефолта заёмщика и показатели портфеля")
CURRENT = load_current()
MODEL = joblib.load(MODELS_DIR / f"model_{CURRENT['version']}.joblib")


class Client(BaseModel):
    ID: int = Field(gt=0, description="идентификатор клиента")
    LIMIT_BAL: float = Field(gt=0, le=1000000, description="кредитный лимит, NT$")
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6, description="коды 0, 5, 6 редкие")
    MARRIAGE: int = Field(ge=0, le=3, description="код 0 редкий")
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Field(ge=-2, le=8, description="статус погашения, сентябрь")
    PAY_2: int = Field(ge=-2, le=8)
    PAY_3: int = Field(ge=-2, le=8)
    PAY_4: int = Field(ge=-2, le=8)
    PAY_5: int = Field(ge=-2, le=8)
    PAY_6: int = Field(ge=-2, le=8)
    BILL_AMT1: float = Field(description="сумма счёта за сентябрь, NT$")
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0, description="платёж за сентябрь, NT$")
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


@app.post("/predict")
def predict(client: Client) -> dict:
    X = pd.DataFrame([{k: v for k, v in client.model_dump().items() if k != "ID"}])
    proba = float(MODEL.predict_proba(X)[0, 1])
    return {"ID": client.ID, "probability": round(proba, 3),
            "at_risk": bool(proba >= CURRENT["threshold"]),
            "threshold": CURRENT["threshold"],
            "model_version": CURRENT["version"]}


@app.get("/metrics")
def metrics() -> dict:
    if not DB_PATH.exists():
        return {"batch": None, "note": "порции еще не обработаны"}
    con = connect(DB_PATH)
    hist = metrics_table(con)
    con.close()
    if hist.empty:
        return {"batch": None, "note": "порции еще не обработаны"}
    cur = hist.iloc[-1]
    return {"batch": hist.index[-1],
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in cur.items()}}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": CURRENT["version"],
            "threshold": CURRENT["threshold"]}
