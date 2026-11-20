"""ПР5. Замер времени ответа программного интерфейса (требование N-02).

Отправляет 200 последовательных запросов /predict через программный
интерфейс (в процессе, без сетевых задержек) и сохраняет среднее и 95-й
перцентиль времени ответа в reports/pr5/api_timing.json.

Запуск из корня репозитория:
    python src/api_benchmark.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from config import REPORTS_DIR

OUT = REPORTS_DIR / "pr5"
N_REQUESTS = 200


def main() -> None:
    from app.api import app  # noqa: PLC0415 (импорт после настройки пути)
    client = TestClient(app)
    row = {"ID": 1, "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1,
           "AGE": 30}
    row.update({f"PAY_{i}": 0 for i in range(1, 7)})
    row.update({f"BILL_AMT{i}": 20000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 2000 for i in range(1, 7)})

    times = []
    for k in range(N_REQUESTS):
        t0 = time.perf_counter()
        response = client.post("/predict", json={**row, "ID": k + 1})
        times.append((time.perf_counter() - t0) * 1000)
        assert response.status_code == 200

    ordered = sorted(times)
    result = {"n_requests": N_REQUESTS,
              "mean_ms": round(sum(times) / len(times), 1),
              "p95_ms": round(ordered[int(0.95 * len(ordered))], 1),
              "max_ms": round(max(times), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "api_timing.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Запросов: {N_REQUESTS}; среднее время ответа {result['mean_ms']} мс; "
          f"95-й перцентиль {result['p95_ms']} мс; максимум {result['max_ms']} мс")


if __name__ == "__main__":
    main()
