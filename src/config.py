"""Общие настройки проекта: пути, начальное значение генератора, состав столбцов.

Пути строятся от корня репозитория, поэтому скрипты можно запускать
из любого каталога.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "credit_default.csv"   # исходный набор, не изменяется
INITIAL_PATH = DATA_DIR / "initial.csv"              # исходная часть (ПР2)
INCOMING_DIR = DATA_DIR / "incoming"                 # порции «новых» данных (ПР2)

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42

ID_COL = "ID"
TARGET = "dpnm"  # дефолт в следующем месяце: 1 — да, 0 — нет

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]          # статус погашения, сентябрь … апрель
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]     # сумма счёта, сентябрь … апрель
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]   # сумма платежа, сентябрь … апрель
