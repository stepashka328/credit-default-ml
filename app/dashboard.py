"""ПР5. Панель показателей (Streamlit).

Читает данные из хранилища data/credit_system.db, поэтому после обработки
новой порции показатели обновляются без изменения кода. Структура:
  - строка карточек с текущими значениями ключевых показателей и их
    изменением относительно предыдущей порции;
  - блок предупреждений о нарушении порогов мониторинга;
  - графики динамики показателей и качества модели;
  - таблица клиентов последней порции с наибольшим риском дефолта.

Запуск из корня репозитория:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import DB_PATH, ID_COL, TARGET  # noqa: E402
from monitoring import evaluate_alerts  # noqa: E402
from storage import connect, metrics_table, runs_table  # noqa: E402

st.set_page_config(page_title="Система анализа риска дефолта",
                   page_icon="📊", layout="wide")

st.title("Система анализа риска дефолта заёмщиков")
st.caption("вероятность дефолта в следующем месяце; порции обновляются командой "
           "python src/process_batch.py <файл порции>")


def load_data():
    con = connect(DB_PATH)
    hist = metrics_table(con)
    runs = runs_table(con)
    top = pd.read_sql(
        "SELECT r.ID, r.probability, r.at_risk, d.LIMIT_BAL, d.PAY_1, "
        "       d.AGE, d.bill_avg, r.model_version "
        "FROM (SELECT * FROM predictions WHERE batch = "
        "      (SELECT batch FROM predictions ORDER BY rowid DESC LIMIT 1)) r "
        "LEFT JOIN (SELECT batch, ID, LIMIT_BAL, PAY_1, AGE, "
        "                  AVG(BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + "
        "                      BILL_AMT5 + BILL_AMT6) / 6.0 AS bill_avg "
        "          FROM raw_data GROUP BY batch, ID) d "
        "USING (batch, ID) ORDER BY r.probability DESC LIMIT 15", con)
    con.close()
    return hist, runs, top


if not DB_PATH.exists():
    st.info("Хранилище пусто: обработайте первую порцию командой "
            "python src/process_batch.py data/incoming/batch_01.csv")
    st.stop()

hist, runs, top = load_data()
if hist.empty:
    st.info("Хранилище пусто: обработайте первую порцию командой "
            "python src/process_batch.py data/incoming/batch_01.csv")
    st.stop()

cur, prev = hist.iloc[-1], (hist.iloc[-2] if len(hist) > 1 else None)
last_run = runs[runs.batch == hist.index[-1]].iloc[-1] if (runs.batch == hist.index[-1]).any() else None
st.markdown(f"**Последняя порция:** {hist.index[-1]} · "
            f"модель **{top.model_version.iloc[0]}** · "
            f"{last_run.status if last_run is not None else ''} "
            f"{last_run.rows if last_run is not None else ''} записей"
            f" за {last_run.duration_s:.1f} с" if last_run is not None else "")

# ---- строка карточек ключевых показателей ----
c = st.columns(5)


def card(col, label, value, delta=None, inverse=False, fmt="{:.1%}"):
    if delta is not None:
        col.metric(label, fmt.format(value),
                   f"{delta * 100:+.1f} п. п.",
                   delta_color="inverse" if inverse else "normal")
    else:
        col.metric(label, fmt.format(value))


card(c[0], "Доля в зоне риска", cur.risk_share,
     None if prev is None else cur.risk_share - prev.risk_share, inverse=True)
c[1].metric("Ожидаемые потери, млн NT$",
            f"{cur.expected_loss_kntd / 1000:.1f}".replace(".", ","),
            None if prev is None else
            f"{(cur.expected_loss_kntd - prev.expected_loss_kntd) / 1000:+.1f}".replace(".", ","))
card(c[2], "Средняя вероятность дефолта", cur.mean_pd,
     None if prev is None else cur.mean_pd - prev.mean_pd, inverse=True)
c[3].metric("Клиентов в порции", f"{int(cur.n_clients):,}".replace(",", " "),
            None if prev is None else f"{cur.n_clients - prev.n_clients:+,.0f}".replace(",", " "))
c[4].metric("Доля отклонённых записей", f"{cur.rejected_share:.1%}",
            None if prev is None else f"{(cur.rejected_share - prev.rejected_share) * 100:+.1f} п. п.")

# ---- предупреждения ----
alerts = evaluate_alerts(cur.to_dict(), None if prev is None else prev.to_dict())
warnings_ = [a["text"] for a in alerts if a["level"] == "warning"]
infos = [a["text"] for a in alerts if a["level"] == "info"]
for text in warnings_:
    st.warning("⚠ " + text)
for text in infos:
    st.info(text)
if not warnings_ and not infos:
    st.success("Нарушений порогов мониторинга нет")

# ---- динамика показателей ----
hist_reset = hist.reset_index().rename(columns={"index": "batch"})

left, right = st.columns(2)
with left:
    st.markdown("**Доля зоны риска, фактическая и прогнозная доля дефолтов**")
    rows = hist_reset.melt(id_vars=["batch"],
                           value_vars=["risk_share", "mean_pd", "actual_rate"],
                           var_name="показатель", value_name="value")
    labels = {"risk_share": "доля в зоне риска", "mean_pd": "средняя прогнозная PD",
              "actual_rate": "фактическая доля дефолтов"}
    rows["показатель"] = rows["показатель"].map(labels)
    st.altair_chart(
        alt.Chart(rows).mark_line(point=True).encode(
            x=alt.X("batch:N", title="порция"),
            y=alt.Y("value:Q", title="доля",
                    scale=alt.Scale(domain=[0, 0.45])),
            color=alt.Color("показатель:N", title="")).properties(height=260),
        use_container_width=True)
with right:
    st.markdown("**Качество модели и стабильность данных по порциям**")
    psi_cols = [c for c in hist_reset.columns if c.startswith("psi_")]
    psi_rows = hist_reset.melt(id_vars=["batch"], value_vars=psi_cols,
                               var_name="признак", value_name="psi")
    psi_rows["признак"] = psi_rows["признак"].str.replace("psi_", "", regex=False)
    rule_psi = pd.DataFrame({"y": [0.25]})
    psi_rule = alt.Chart(rule_psi).mark_rule(color="#b3261e", strokeDash=[4, 2]).encode(y="y")
    psi_chart = alt.Chart(psi_rows).mark_line(point=True).encode(
        x=alt.X("batch:N", title="порция"),
        y=alt.Y("psi:Q", title="PSI ключевых признаков"),
        color=alt.Color("признак:N", title="")).properties(height=200)
    roc_chart = alt.Chart(hist_reset).mark_line(point=True, color="#2a78d6").encode(
        x=alt.X("batch:N", title="порция"),
        y=alt.Y("roc_auc:Q", title="ROC-AUC", scale=alt.Scale(domain=[0.70, 0.82])))
    st.altair_chart((psi_chart + psi_rule).properties(height=200), use_container_width=True)
    st.altair_chart(roc_chart.properties(height=200), use_container_width=True)

# ---- клиенты с наибольшим риском ----
st.markdown("**Клиенты последней порции с наибольшим риском дефолта** "
            "(приоритет ручной проверки и пересмотра лимитов)")
show = top.rename(columns={"probability": "вероятность дефолта", "at_risk": "зона риска",
                           "LIMIT_BAL": "лимит, NT$", "PAY_1": "просрочка, мес.",
                           "AGE": "возраст", "bill_avg": "средний счёт, NT$"})
show.insert(0, "ID", show.pop("ID"))
st.dataframe(show, use_container_width=True, height=420)

# ---- журнал запусков ----
with st.expander("Журнал запусков"):
    st.dataframe(runs, use_container_width=True)
