"""ПР5. Хранилище системы: база данных SQLite data/credit_system.db.

Четыре группы данных (пособие, п. 5.2.2):
  raw_data         - принятые записи порций (очищенные);
  predictions      - вероятности дефолта и метки зоны риска с версией модели;
  metrics_history  - история ключевых показателей: одна запись на порцию
                    и показатель (доля зоны риска, ожидаемые потери, PSI...);
  runs             - журнал запусков (порция, статус, записи, версия модели,
                    длительность, сообщение).

Идемпотентность обеспечивается первичными ключами (порция + клиент,
порция + показатель) и командой INSERT OR REPLACE: повторная обработка той
же порции перезаписывает результаты, не создавая дубликатов.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from config import ID_COL, TARGET
from features import BILL_COLS, PAY_AMT_COLS, PAY_COLS, RAW_FEATURES

RAW_COLS = [ID_COL] + RAW_FEATURES + [TARGET]


def connect(db_path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db(con: sqlite3.Connection) -> None:
    cols = ", ".join(
        [f"{c} {'INTEGER' if c in (ID_COL, TARGET, 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE')
                or c in PAY_COLS else 'REAL'}" for c in RAW_COLS])
    con.executescript(f"""
    CREATE TABLE IF NOT EXISTS raw_data (
        batch TEXT, {cols}, PRIMARY KEY (batch, {ID_COL}));
    CREATE TABLE IF NOT EXISTS predictions (
        batch TEXT, {ID_COL} INTEGER, probability REAL, at_risk INTEGER,
        model_version TEXT, PRIMARY KEY (batch, {ID_COL}));
    CREATE TABLE IF NOT EXISTS metrics_history (
        batch TEXT, processed_at TEXT, metric TEXT, value REAL,
        PRIMARY KEY (batch, metric));
    CREATE TABLE IF NOT EXISTS runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT, batch TEXT, started_at TEXT,
        status TEXT, rows INTEGER, duration_s REAL, model_version TEXT,
        message TEXT);
    """)
    con.commit()


def save_raw(con, df: pd.DataFrame, batch: str) -> None:
    """Принятые записи порции (перезапись без дубликатов)."""
    data = df.copy()
    data.insert(0, "batch", batch)
    placeholders = ", ".join("?" * (len(RAW_COLS) + 1))
    con.executemany(
        f"INSERT OR REPLACE INTO raw_data (batch, {', '.join(RAW_COLS)}) "
        f"VALUES ({placeholders})",
        data[["batch"] + RAW_COLS].itertuples(index=False, name=None))
    con.commit()


def save_predictions(con, batch: str, ids, proba, at_risk, version: str) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO predictions "
        "(batch, ID, probability, at_risk, model_version) VALUES (?, ?, ?, ?, ?)",
        [(batch, int(i), float(p), int(a), version)
         for i, p, a in zip(ids, proba, at_risk)])
    con.commit()


def save_metrics(con, batch: str, metrics: dict, processed_at: str) -> None:
    con.executemany(
        "INSERT OR REPLACE INTO metrics_history "
        "(batch, processed_at, metric, value) VALUES (?, ?, ?, ?)",
        [(batch, processed_at, k, float(v)) for k, v in metrics.items()])
    con.commit()


def log_run(con, batch: str, started_at: str, status: str, rows: int,
            duration_s: float, version: str, message: str) -> int:
    cur = con.execute(
        "INSERT INTO runs (batch, started_at, status, rows, duration_s, "
        "model_version, message) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (batch, started_at, status, rows, duration_s, version, message))
    con.commit()
    return cur.lastrowid


def metrics_table(con) -> pd.DataFrame:
    """История показателей: порции x показатели (значения)."""
    return pd.read_sql("SELECT batch, processed_at, metric, value "
                       "FROM metrics_history ORDER BY rowid",
                       con).pivot(index="batch", columns="metric",
                                  values="value")


def latest_metrics(con) -> dict | None:
    hist = metrics_table(con)
    return None if hist.empty else hist.iloc[-1].to_dict()


def previous_metrics(con) -> dict | None:
    hist = metrics_table(con)
    return None if len(hist) < 2 else hist.iloc[-2].to_dict()


def runs_table(con) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM runs ORDER BY run_id", con)
