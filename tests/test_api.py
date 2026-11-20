"""Тесты программного интерфейса (ПР5, app/api.py).

Запуск из корня репозитория:
    python -m pytest tests
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.api import app  # noqa: E402

client = TestClient(app)


def _client_row(**overrides) -> dict:
    row = {"ID": 1, "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1,
           "AGE": 30}
    row.update({f"PAY_{i}": 0 for i in range(1, 7)})
    row.update({f"BILL_AMT{i}": 20000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 2000 for i in range(1, 7)})
    row.update(overrides)
    return row


def test_health():
    """Состояние сервиса: версия модели и порог присутствуют."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"].startswith("v")
    assert 0 < body["threshold"] < 1


def test_predict_valid():
    """Корректный запрос: вероятность от 0 до 1 и метка зоны риска."""
    response = client.post("/predict", json=_client_row(PAY_1=2, BILL_AMT1=50000))
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["at_risk"], bool)
    assert body["model_version"].startswith("v")


def test_predict_risk_order():
    """Клиент с просрочкой получает более высокую вероятность риска."""
    risky = client.post("/predict", json=_client_row(PAY_1=2)).json()["probability"]
    calm = client.post("/predict", json=_client_row(PAY_1=-1, LIMIT_BAL=500000,
                                                     PAY_AMT1=40000)).json()["probability"]
    assert risky > calm


def test_predict_invalid_rejected():
    """Недопустимые значения отклоняются с кодом 422 до обращения к модели."""
    response = client.post("/predict", json=_client_row(PAY_1=12, AGE=150))
    assert response.status_code == 422
    assert "PAY_1" in response.text or "AGE" in response.text


def test_metrics():
    """Показатели: метод отвечает либо данными порции, либо сообщением о пустом хранилище."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
