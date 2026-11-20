"""ПР4. Проверки качества данных: требования Q1–Q10 из ПР2 (таблица 5).

Каждая проверка автоматически выполняется для любой порции и возвращает
число нарушений; требования делятся на блокирующие (порция не обрабатывается)
и предупреждающие (фиксируются в отчете). Действия на уровне записи
(исключение, перекодировка) выполняет функция clean модуля src/features.py.

Используется в src/evaluate.py (ПР4) и src/process_batch.py (ПР5):
    from quality import check_quality
    report = check_quality(pd.read_csv("data/incoming/batch_01.csv"))
    if not report["ok"]:
        ...  # порция отклоняется, причины - в report["failed"]
"""

import pandas as pd

from config import ID_COL, PAY_AMT_COLS, PAY_COLS
from features import RAW_FEATURES

SCHEMA_COLS = [ID_COL] + RAW_FEATURES          # обязательные столбцы порции
EDU_VALID, EDU_RECODE = {1, 2, 3, 4}, {0, 5, 6}
MAR_VALID, MAR_RECODE = {1, 2, 3}, {0}
REJECT_SHARE_T = 0.05                          # Q10: доля отклонённых записей


def _check(qid, name, violations, action, blocking):
    return {"id": qid, "name": name, "violations": int(violations),
            "action": action, "blocking": bool(blocking),
            "failed": bool(violations)}


def check_quality(df: pd.DataFrame) -> dict:
    """Проверка порции по требованиям Q1–Q10. Возвращает отчет:

    ok            - нет нарушений блокирующих проверок;
    checks        - список проверок с числом нарушений и действием;
    rejected_share- доля записей, которые будут исключены (Q3, Q4, Q7, Q8, Q9);
    failed        - названия нарушенных блокирующих проверок.
    """
    n = len(df)
    checks = []

    # Q1: идентификатор не пуст и уникален (блокирующее)
    checks.append(_check(
        "Q1", "ID не пуст и уникален в порции",
        df[ID_COL].isna().sum() + df[ID_COL].duplicated().sum(),
        "порция не обрабатывается", True))

    # Q2: набор столбцов, типы и отсутствие пропусков (блокирующее)
    missing_cols = [c for c in SCHEMA_COLS if c not in df.columns]
    n_missing = int(sum(df[c].isna().sum() for c in SCHEMA_COLS if c in df.columns))
    not_numeric = [c for c in SCHEMA_COLS
                   if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    violations = n_missing + len(missing_cols) + len(not_numeric)
    checks.append(_check(
        "Q2", "набор столбцов, числовые типы и отсутствие пропусков",
        violations, "порция не обрабатывается", True))

    # Q3–Q4, Q7–Q8: недопустимые значения - запись исключается
    checks.append(_check(
        "Q3", "LIMIT_BAL больше 0",
        int((df["LIMIT_BAL"] <= 0).sum()) if "LIMIT_BAL" in df else 0,
        "запись исключается", False))
    checks.append(_check(
        "Q4", "AGE в диапазоне 18..100",
        int((~df["AGE"].between(18, 100)).sum()) if "AGE" in df else 0,
        "запись исключается", False))
    checks.append(_check(
        "Q5", "EDUCATION из 0..6, коды 0/5/6 перекодируются в 4",
        int((~df["EDUCATION"].isin(EDU_VALID)).sum()) if "EDUCATION" in df else 0,
        "перекодировка", False))
    checks.append(_check(
        "Q6", "MARRIAGE из 0..3, код 0 перекодируется в 3",
        int((~df["MARRIAGE"].isin(MAR_VALID)).sum()) if "MARRIAGE" in df else 0,
        "перекодировка", False))
    pay_bad = sum(int((~df[c].isin(range(-2, 9))).sum())
                 for c in PAY_COLS if c in df.columns) if all(c in df.columns for c in PAY_COLS) else 0
    checks.append(_check(
        "Q7", "PAY_1..PAY_6 - целые значения от -2 до 8",
        pay_bad, "запись исключается", False))
    payamt_bad = int(sum((df[c] < 0).sum() for c in PAY_AMT_COLS)) \
        if all(c in df.columns for c in PAY_AMT_COLS) else 0
    checks.append(_check(
        "Q8", "PAY_AMT1..PAY_AMT6 не меньше 0",
        payamt_bad, "запись исключается", False))

    # Q9: полные дубликаты строк (без ID) удаляются
    feature_cols = [c for c in SCHEMA_COLS if c in df.columns]
    checks.append(_check(
        "Q9", "нет полных дубликатов строк (без ID)",
        int(df.duplicated(subset=feature_cols).sum()),
        "дубликаты удаляются", False))

    # доля записей, затронутых исключением и дубликатами (Q10, предупреждающее)
    drop_mask = pd.Series(False, index=df.index)
    if "LIMIT_BAL" in df:
        drop_mask |= df["LIMIT_BAL"].isna() | (df["LIMIT_BAL"] <= 0)
    if "AGE" in df:
        drop_mask |= df["AGE"].isna() | ~df["AGE"].between(18, 100)
    if all(c in df.columns for c in PAY_COLS):
        drop_mask |= df[PAY_COLS].isna().any(axis=1) \
            | (~df[PAY_COLS].isin(range(-2, 9))).any(axis=1)
    if all(c in df.columns for c in PAY_AMT_COLS):
        drop_mask |= df[PAY_AMT_COLS].isna().any(axis=1) | (df[PAY_AMT_COLS] < 0).any(axis=1)
    if feature_cols:
        drop_mask |= df.duplicated(subset=feature_cols)
    rejected_share = float(drop_mask.mean()) if n else 0.0
    checks.append(_check(
        "Q10", "доля отклонённых записей не более 5 %",
        int(rejected_share > REJECT_SHARE_T),
        "порция помечается «на проверку»", False))

    failed = [c["id"] + " " + c["name"] for c in checks if c["failed"] and c["blocking"]]
    return {"ok": not failed, "rows": n, "checks": checks,
            "rejected_share": round(rejected_share, 4), "failed": failed}


def format_report(report: dict) -> str:
    """Текстовое представление отчета о проверке (для журнала и отчетов)."""
    lines = [f"проверка качества: {report['rows']} записей, "
             f"{'пройдена' if report['ok'] else 'НЕ пройдена'}"]
    for c in report["checks"]:
        if c["failed"]:
            lines.append(f"  {c['id']}: {c['name']} - нарушений {c['violations']} "
                         f"({c['action']})")
    if report["rejected_share"] > 0:
        lines.append(f"  доля отклонённых записей: "
                     f"{report['rejected_share']:.1%}")
    return "; ".join(lines)
