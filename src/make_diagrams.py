"""ПР5. Диаграммы: обновлённая диаграмма контейнеров C4 и диаграмма
развертывания UML.

Для каждой диаграммы создаются два файла:
  diagrams/pr5/c4_containers_pr5.drawio, uml_deployment.drawio
      - исходники draw.io: редактируются в diagrams.net и хранятся
        в репозитории вместе с кодом;
  diagrams/pr5/c4_containers_pr5.png, uml_deployment.png
      - изображения для отчета (рендер Playwright + SVG).

Запуск из корня репозитория:
    python src/make_diagrams.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagrams" / "pr5"

FONT = "'Noto Sans', 'DejaVu Sans', 'Segoe UI', system-ui, sans-serif"
INK, SUB = "#0f172a", "#475569"
NODE_BG, NODE_BORDER = "#EFF6FF", "#2a78d6"      # программы-интерфейсы
STORE_BG, STORE_BORDER = "#F0FDF4", "#1baf7a"     # хранилища и артефакты
PROC_BG, PROC_BORDER = "#FFF7ED", "#d97706"       # процессы и скрипты
EXT_BG, EXT_BORDER = "#F5F3FF", "#8b5cf6"         # внешние узлы
BOUNDARY, EDGE = "#94a3b8", "#94a3b8"


def svg_text(x, y, text, size=12, fill=SUB, weight="", anchor="middle"):
    weight = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" '
            f'font-size="{size}"{weight} font-family="{FONT}">{text}</text>')


def node(x, y, w, h, title, desc, bg, border, r=10):
    parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
             f'fill="{bg}" stroke="{border}" stroke-width="2"/>',
             svg_text(x + w / 2, y + 26, title, 15, INK, "600")]
    parts += [svg_text(x + w / 2, y + 46 + i * 16, line, 11.5, SUB)
              for i, line in enumerate(desc)]
    return "".join(parts)


def edge(points, label="", lx=None, ly=None):
    pts = " ".join(f"{px},{py}" for px, py in points)
    out = [f'<polyline points="{pts}" fill="none" stroke="{EDGE}" '
           f'stroke-width="1.8" marker-end="url(#a)"/>']
    if label and lx is not None:
        out.append(f'<rect x="{lx - len(label) * 3.7}" y="{ly - 12}" '
                   f'width="{len(label) * 7.4 + 6}" height="17" fill="#ffffff" '
                   f'opacity="0.94" rx="3"/>')
        out.append(svg_text(lx, ly, label, 11.5, SUB))
    return "".join(out)


def page(inner, width, height, caption):
    return f"""<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#fff; font-family:{FONT}; }}
    #root {{ width:{width}px; padding:22px 14px; }}
    .cap {{ text-align:center; font-size:12px; color:{SUB}; margin-top:10px; }}
    </style></head><body><div id="root">
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5"
    orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="{EDGE}"/></marker></defs>
    {inner}</svg><div class="cap">{caption}</div></div></body></html>"""


# ------------------------------------------------------------------ C4
def containers_inner() -> str:
    p = []
    p.append(svg_text(780, 40, "Диаграмма контейнеров C4 (обновлена в ПР5)", 20, INK, "700"))
    p.append(f'<rect x="400" y="70" width="700" height="710" rx="6" fill="#ffffff" '
             f'stroke="{BOUNDARY}" stroke-width="1.6" stroke-dasharray="8,5"/>')
    p.append(svg_text(750, 96, "Система анализа риска дефолта (credit-default-ml)", 13, SUB, "600"))

    # акторы и внешние системы
    p.append(node(60, 140, 190, 74, "Риск-аналитик", ["просматривает панель,"], EXT_BG, EXT_BORDER, 24))
    p.append(node(60, 690, 190, 74, "Внешняя система", ["запрашивает прогнозы"], EXT_BG, EXT_BORDER, 24))
    p.append(node(1150, 120, 210, 84, "Специалист по ML", ["запускает обработку порций,", "переобучение и откат версий"], EXT_BG, EXT_BORDER, 24))
    p.append(node(1150, 660, 210, 84, "Источник данных", ["ежемесячные выгрузки", "клиентов (CSV)"], EXT_BG, EXT_BORDER, 24))

    # контейнеры системы
    p.append(node(430, 130, 180, 100, "Панель", ["app/dashboard.py", "Streamlit, порт 8501"], NODE_BG, NODE_BORDER))
    p.append(node(690, 130, 180, 100, "Хранилище", ["data/credit_system.db", "SQLite: порции,", "предсказания, показатели,", "журнал запусков"], STORE_BG, STORE_BORDER))
    p.append(node(940, 130, 150, 100, "Сценарий", ["src/process_batch.py", "проверка, признаки,", "предсказание,", "показатели"], PROC_BG, PROC_BORDER))
    p.append(node(430, 400, 180, 100, "API", ["app/api.py", "FastAPI + uvicorn,", "порт 8000"], NODE_BG, NODE_BORDER))
    p.append(node(690, 400, 180, 100, "Файлы моделей", ["models/model_vN.joblib,", "current.json - указатель", "на действующую версию"], STORE_BG, STORE_BORDER))
    p.append(node(940, 400, 150, 100, "Переобучение", ["src/retrain.py", "претендент, замена,", "откат версии"], PROC_BG, PROC_BORDER))

    # связи с акторами и внешними системами
    p.append(edge([(250, 177), (430, 177)], "HTTP :8501", 340, 168))
    p.append(edge([(250, 727), (330, 727), (330, 450), (430, 450)],
                  "HTTP JSON :8000", 300, 718))
    # специалист по ML: запуск сценария и переобучения (обход снизу, без пересечений)
    p.append(edge([(1150, 162), (1090, 162)]))
    p.append(edge([(1360, 162), (1400, 162), (1400, 755), (1015, 755), (1015, 500)],
                  "по сигналу панели", 1250, 772))
    # источник данных: файл порции подается в сценарий (правый канал)
    p.append(edge([(1150, 702), (1110, 702), (1110, 176), (1090, 176)],
                  "файл порции CSV", 1110, 600))

    # внутренние связи контейнеров
    p.append(edge([(610, 180), (690, 180)], "SQL", 650, 172))
    p.append(edge([(520, 400), (520, 330), (730, 330), (730, 230)], "SQL", 625, 322))
    p.append(edge([(610, 450), (690, 450)], "joblib", 650, 442))
    p.append(edge([(940, 180), (870, 180)], "SQLite", 905, 172))
    p.append(edge([(1015, 230), (1015, 345), (855, 345), (855, 400)], "joblib", 935, 337))
    p.append(edge([(1015, 400), (1015, 390), (750, 390), (750, 230)],
                  "чтение порций", 890, 382))
    p.append(edge([(940, 450), (870, 450)], "запись", 905, 432))
    return "".join(p)


# ------------------------------------------------------------------ deployment
def deployment_inner() -> str:
    p = []
    p.append(svg_text(670, 40, "Диаграмма развертывания UML", 20, INK, "700"))
    # узел - рабочая станция
    p.append(f'<rect x="330" y="80" width="660" height="600" rx="10" fill="#ffffff" '
             f'stroke="#64748b" stroke-width="2.4"/>')
    p.append(f'<rect x="330" y="80" width="660" height="54" rx="10" fill="#F8FAFC" '
             f'stroke="#64748b" stroke-width="2.4"/>')
    p.append(f'<rect x="330" y="106" width="660" height="28" fill="#F8FAFC"/>')
    p.append(svg_text(660, 102, "«device» Рабочая станция специалиста ML", 15, INK, "600"))
    p.append(svg_text(660, 126, "Windows/Linux; Python 3.12 (venv), Git, requirements.txt", 12, SUB))

    # среда выполнения (процессы)
    p.append(f'<rect x="360" y="160" width="600" height="330" rx="8" fill="#ffffff" '
             f'stroke="{BOUNDARY}" stroke-width="1.5" stroke-dasharray="7,4"/>')
    p.append(svg_text(660, 186, "«execution environment» Python 3.12 (venv) - процессы", 12.5, SUB, "600"))
    p.append(node(390, 210, 250, 110, "Терминал (CLI)",
                  ["python src/process_batch.py", "data/incoming/batch_XX.csv", "python src/retrain.py"],
                  PROC_BG, PROC_BORDER))
    p.append(node(700, 210, 230, 110, "Процесс streamlit",
                  ["streamlit run app/dashboard.py", "порт 8501 (localhost)"], PROC_BG, PROC_BORDER))
    p.append(node(390, 370, 250, 110, "Процесс uvicorn",
                  ["uvicorn app.api:app", "порт 8000 (localhost)"], PROC_BG, PROC_BORDER))
    p.append(node(700, 370, 230, 110, "Скрипты Python",
                  ["src/quality.py, monitoring.py,", "storage.py, features.py;", "tests/ (pytest)"],
                  NODE_BG, NODE_BORDER))

    # артефакты - файлы на диске узла
    p.append(node(390, 520, 250, 110, "Хранилище",
                  ["data/credit_system.db", "(SQLite, WAL)"], STORE_BG, STORE_BORDER))
    p.append(node(660, 520, 140, 110, "Модели",
                  ["models/model_vN", ".joblib,", "current.json"], STORE_BG, STORE_BORDER))
    p.append(node(820, 520, 110, 110, "Порции",
                  ["data/incoming/", "batch_XX.csv"], STORE_BG, STORE_BORDER))

    # внешние узлы
    p.append(node(60, 200, 200, 84, "Браузер", ["риск-аналитика", "(HTTP localhost:8501)"], EXT_BG, EXT_BORDER, 24))
    p.append(node(60, 610, 200, 84, "Внешняя система", ["маркетинговых кампаний", "(HTTP JSON :8000)"], EXT_BG, EXT_BORDER, 24))
    p.append(node(1060, 560, 210, 84, "Источник данных", ["ежемесячные выгрузки", "порций (CSV)"], EXT_BG, EXT_BORDER, 24))

    # связи между узлами
    p.append(edge([(260, 242), (330, 242)], "HTTP :8501", 295, 233))
    p.append(edge([(260, 652), (330, 652)], "HTTP JSON :8000", 295, 643))
    p.append(edge([(1060, 600), (990, 600)], "файл порции", 1025, 591))

    # связи внутри узла: процесс - файл
    p.append(edge([(640, 290), (648, 290), (648, 555), (640, 555)], "SQLite", 648, 505))
    p.append(edge([(640, 425), (665, 425), (665, 490), (690, 490), (690, 520)],
                  "joblib", 740, 510))
    return "".join(p)


# ------------------------------------------------------------------ draw.io
def _v(cid, x, y, w, h, label, style):
    return (f'<mxCell id="{cid}" value="{label}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def _e(eid, src, tgt, label=""):
    style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=1.8;"
             "strokeColor=#475569;fontColor=#475569;fontSize=11;endArrow=blockThin;"
             "endFill=1;endSize=6;labelBackgroundColor=#ffffff;")
    return (f'<mxCell id="{eid}" value="{label}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')


BOX = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;"
ST_NODE, ST_STORE = BOX + "fillColor=#EFF6FF;strokeColor=#2a78d6;", BOX + "fillColor=#F0FDF4;strokeColor=#1baf7a;"
ST_PROC, ST_EXT = BOX + "fillColor=#FFF7ED;strokeColor=#d97706;", BOX + "fillColor=#F5F3FF;strokeColor=#8b5cf6;"
ST_SYS = "rounded=0;dashed=1;html=1;fillColor=none;strokeColor=#94a3b8;verticalAlign=top;fontStyle=1;"
ST_DEV = "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#64748b;strokeWidth=2.4;verticalAlign=top;fontSize=14;fontStyle=1;"
ST_ENV = "rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#94a3b8;dashed=1;verticalAlign=top;fontSize=12;fontStyle=1;"
B, BR = "&lt;b&gt;", "&lt;/b&gt;"


def make_drawio_containers() -> str:
    cells = [
        _v("sys", 400, 70, 700, 710, "Система анализа риска дефолта (credit-default-ml)", ST_SYS),
        _v("a1", 60, 140, 190, 74, f"{B}Риск-аналитик{BR}&lt;br&gt;просматривает панель", ST_EXT),
        _v("a2", 60, 690, 190, 74, f"{B}Внешняя система{BR}&lt;br&gt;запрашивает прогнозы", ST_EXT),
        _v("a3", 1150, 120, 210, 84, f"{B}Специалист по ML{BR}&lt;br&gt;запускает обработку порций,&lt;br&gt;переобучение и откат версий", ST_EXT),
        _v("a4", 1150, 660, 210, 84, f"{B}Источник данных{BR}&lt;br&gt;ежемесячные выгрузки клиентов (CSV)", ST_EXT),
        _v("n1", 430, 130, 180, 100, f"{B}Панель{BR}&lt;br&gt;app/dashboard.py&lt;br&gt;Streamlit, порт 8501", ST_NODE),
        _v("s1", 690, 130, 180, 100, f"{B}Хранилище{BR}&lt;br&gt;data/credit_system.db (SQLite)&lt;br&gt;порции, предсказания,&lt;br&gt;показатели, журнал", ST_STORE),
        _v("p1", 940, 130, 150, 100, f"{B}Сценарий{BR}&lt;br&gt;src/process_batch.py&lt;br&gt;проверка, признаки,&lt;br&gt;предсказание, показатели", ST_PROC),
        _v("n2", 430, 400, 180, 100, f"{B}API{BR}&lt;br&gt;app/api.py&lt;br&gt;FastAPI + uvicorn, порт 8000", ST_NODE),
        _v("s2", 690, 400, 180, 100, f"{B}Файлы моделей{BR}&lt;br&gt;models/model_vN.joblib,&lt;br&gt;current.json - указатель&lt;br&gt;на действующую версию", ST_STORE),
        _v("p2", 940, 400, 150, 100, f"{B}Переобучение{BR}&lt;br&gt;src/retrain.py&lt;br&gt;претендент, замена, откат", ST_PROC),
        _e("e1", "a1", "n1", "HTTP :8501"), _e("e2", "a2", "n2", "HTTP JSON :8000"),
        _e("e3", "a3", "p1", "запуск"), _e("e4", "a3", "p2", "по сигналу панели"),
        _e("e5", "a4", "p1", "файл порции CSV"),
        _e("e6", "n1", "s1", "SQL"), _e("e7", "n2", "s1", "SQL"),
        _e("e8", "n2", "s2", "joblib"), _e("e9", "p1", "s1", "SQLite"),
        _e("e10", "p1", "s2", "joblib"), _e("e11", "p2", "s1", "чтение порций"),
        _e("e12", "p2", "s2", "запись vN"),
    ]
    return ('<mxfile host="app.diagrams.net" modified="2026-11-15T12:00:00Z" '
            'agent="credit-default-ml"><diagram id="c4-containers-pr5" '
            'name="Контейнеры C4 (ПР5)"><mxGraphModel dx="1400" dy="880" grid="1" '
            'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            'page="1" pageScale="1" pageWidth="1600" pageHeight="950" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells)
            + "</root></mxGraphModel></diagram></mxfile>")


def make_drawio_deployment() -> str:
    cells = [
        _v("ws", 330, 80, 660, 600, "«device» Рабочая станция специалиста ML&lt;br&gt;"
          "Windows/Linux; Python 3.12 (venv), Git, requirements.txt", ST_DEV),
        _v("env", 360, 160, 600, 330, "«execution environment» Python 3.12 (venv) - процессы", ST_ENV),
        _v("c1", 390, 210, 250, 110, f"{B}Терминал (CLI){BR}&lt;br&gt;python src/process_batch.py&lt;br&gt;"
          "data/incoming/batch_XX.csv&lt;br&gt;python src/retrain.py", ST_PROC),
        _v("c2", 700, 210, 230, 110, f"{B}Процесс streamlit{BR}&lt;br&gt;streamlit run app/dashboard.py&lt;br&gt;"
          "порт 8501 (localhost)", ST_PROC),
        _v("c3", 390, 370, 250, 110, f"{B}Процесс uvicorn{BR}&lt;br&gt;uvicorn app.api:app&lt;br&gt;порт 8000 (localhost)", ST_PROC),
        _v("c4", 700, 370, 230, 110, f"{B}Скрипты Python{BR}&lt;br&gt;src/quality.py, monitoring.py,&lt;br&gt;"
          "storage.py, features.py;&lt;br&gt;tests/ (pytest)", ST_NODE),
        _v("f1", 390, 520, 250, 110, f"{B}Хранилище{BR}&lt;br&gt;data/credit_system.db&lt;br&gt;(SQLite, WAL)", ST_STORE),
        _v("f2", 660, 520, 140, 110, f"{B}Модели{BR}&lt;br&gt;models/model_vN.joblib,&lt;br&gt;current.json", ST_STORE),
        _v("f3", 820, 520, 110, 110, f"{B}Порции{BR}&lt;br&gt;data/incoming/&lt;br&gt;batch_XX.csv", ST_STORE),
        _v("b1", 60, 200, 200, 84, f"{B}Браузер риск-аналитика{BR}&lt;br&gt;HTTP localhost:8501", ST_EXT),
        _v("b2", 60, 610, 200, 84, f"{B}Внешняя система кампаний{BR}&lt;br&gt;HTTP JSON :8000", ST_EXT),
        _v("b3", 1060, 560, 210, 84, f"{B}Источник данных{BR}&lt;br&gt;ежемесячные выгрузки (CSV)", ST_EXT),
        _e("d1", "b1", "ws", "HTTP :8501"), _e("d2", "b2", "ws", "HTTP JSON :8000"),
        _e("d3", "b3", "f3", "файл порции CSV"),
        _e("d4", "c1", "f1", "SQLite"), _e("d5", "c3", "f2", "joblib"),
    ]
    return ('<mxfile host="app.diagrams.net" modified="2026-11-15T12:00:00Z" '
            'agent="credit-default-ml"><diagram id="uml-deployment" '
            'name="Развертывание UML (ПР5)"><mxGraphModel dx="1340" dy="760" grid="1" '
            'gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
            'page="1" pageScale="1" pageWidth="1400" pageHeight="800" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells)
            + "</root></mxGraphModel></diagram></mxfile>")


async def render() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = [("c4_containers_pr5", containers_inner(), 1560, 860,
              "Стрелки - обмен данными; пунктирная рамка - граница системы"),
             ("uml_deployment", deployment_inner(), 1340, 760,
              "Один узел - рабочая станция (ограничение N-07); панель и API - процессы "
              "среды Python, хранилище, модели и порции - файлы на диске узла")]
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1500, "height": 1000},
                                      device_scale_factor=2)
        for name, inner, w, h, cap in pages:
            html = page_wrap(inner, w, h, cap)
            path = OUT / f"{name}.html"
            path.write_text(html, encoding="utf-8")
            await page.goto(f"file://{path}", wait_until="networkidle")
            await page.wait_for_timeout(300)
            await page.locator("#root").screenshot(path=str(OUT / f"{name}.png"))
            path.unlink()
            print(f"OK {name}.png")
        await browser.close()
    (OUT / "c4_containers_pr5.drawio").write_text(make_drawio_containers(), encoding="utf-8")
    (OUT / "uml_deployment.drawio").write_text(make_drawio_deployment(), encoding="utf-8")
    print("OK .drawio")


def page_wrap(inner, w, h, cap):
    return page(inner, w, h, cap)


if __name__ == "__main__":
    asyncio.run(render())
