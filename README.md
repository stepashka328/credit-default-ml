# credit-default-ml — прогнозирование дефолта заёмщиков

Семестровый проект по дисциплине «Технологии и инструментарий машинного
обучения» (РТУ МИРЭА, 5 семестр). Тема: «Разработка модели прогнозирования
дефолта заёмщиков методами машинного обучения».

**Задача.** Бинарная классификация: по данным клиента кредитной карты за
апрель – сентябрь 2005 г. оценить вероятность дефолта в следующем месяце.
Система принимает очередную порцию данных, проверяет её качество, применяет
модель и пересчитывает ключевые показатели портфеля (доля клиентов в зоне
риска, ожидаемые потери, средняя вероятность дефолта); панель показателей и
API отображают результаты, мониторинг следит за дрейфом данных и качеством
модели, при сигнале модель переобучается с сохранением версии и откатом.

## Данные

Набор Default of Credit Card Clients (Yeh I.-C., Lien C.-H., 2009),
UCI Machine Learning Repository, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H),
лицензия CC BY 4.0. В проекте используется копия набора с Kaggle, в которой
столбцы названы `PAY_1 … PAY_6`, а целевая переменная — `dpnm`.

| Файл | Записей | Назначение |
|---|---|---|
| `data/raw/credit_default.csv` | 30 000 | исходный набор, не изменяется |
| `data/initial.csv` | 18 000 (60 %) | исходная часть: EDA, обучение, оценка |
| `data/incoming/batch_01.csv … batch_04.csv` | по 3 000 | порции «новых» данных |
| `data/incoming/batch_05.csv` | 3 000 | стресс-порция: имитация дрейфа для демонстрации мониторинга (`src/make_stress_batch.py`, ПР4) |
| `data/demo/bad_batch.csv` | 300 | испорченная порция для демонстрации проверок качества (`src/make_bad_batch.py`) |
| `data/train.csv` | 14 388 | обучающая выборка (80 % исходной части без дубликатов) |
| `data/test.csv` | 3 598 | отложенная тестовая выборка (20 %), используется только в ПР4 |

Даты в наборе нет, поэтому деление случайное, стратифицированное по целевой
переменной, `seed = 42` (`src/split_batches.py`, `src/split_train_test.py`).

## Структура репозитория

```
├── data/
│   ├── raw/                 исходный набор (не изменяется)
│   ├── initial.csv          исходная часть
│   ├── train.csv, test.csv  обучающая и тестовая выборки (ПР3)
│   ├── incoming/            порции новых данных
│   └── processed/           таблицы признаков (создаются, в Git не хранятся)
├── src/
│   ├── config.py            пути, seed, состав столбцов
│   ├── split_batches.py     ПР2: разбиение на исходную часть и порции
│   ├── eda.py               ПР2: разведочный анализ исходной части
│   ├── features.py          ПР3: очистка, новые признаки, пайплайн (Pipeline)
│   ├── split_train_test.py  ПР3: обучающая и тестовая выборки
│   ├── experiments.py       ПР3: эксперименты и выбор модели
│   ├── train.py             ПР3: обучение и сохранение итоговой модели
│   ├── evaluate.py          ПР4: оценка, интерпретация, порции, паспорт модели
│   ├── quality.py           ПР4/ПР5: проверки качества данных Q1–Q10
│   ├── monitoring.py        ПР4/ПР5: PSI и правила мониторинга
│   ├── make_bad_batch.py    ПР4: испорченная порция для проверки Q1–Q10
│   ├── make_stress_batch.py ПР4: стресс-порция batch_05 (имитация дрейфа)
│   ├── storage.py           ПР5: хранилище SQLite (4 таблицы)
│   ├── process_batch.py     ПР5: сценарий обновления одной командой
│   ├── retrain.py           ПР5: переобучение, замена и откат версий
│   └── api_benchmark.py     ПР5: замер времени ответа API (N-02)
├── diagrams/                диаграммы архитектуры (C4, UML; pr5 - draw.io + PNG)
├── reports/                 таблицы и графики, создаваемые скриптами (по работам)
├── models/                  модели model_vN.joblib с описаниями и указателем
│                            действующей версии current.json (порог 0,25)
├── app/                     панель показателей (Streamlit) и API (FastAPI), ПР5
├── tests/                   автоматические тесты (пайплайн и API)
├── requirements.txt         зависимости с точными версиями
└── README.md
```

## Установка

Требуется Python 3.10+ (проверено на 3.12).

```bash
git clone https://github.com/stepashka328/credit-default-ml.git
cd credit-default-ml
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Запуск

Все команды выполняются из корня репозитория.

| Этап | Команда | Результат |
|---|---|---|
| ПР2. Разбиение данных | `python src/split_batches.py` | `data/initial.csv`, `data/incoming/*.csv` |
| ПР2. Разведочный анализ | `python src/eda.py` | `reports/pr2/` |
| ПР3. Обучающая и тестовая выборки | `python src/split_train_test.py` | `data/train.csv`, `data/test.csv` |
| ПР3. Таблица признаков любого файла | `python src/features.py data/incoming/batch_01.csv` | `data/processed/batch_01_features.csv` |
| ПР3. Эксперименты (≈ 3 мин) | `python src/experiments.py` | `reports/experiments.csv`, `reports/pr3/` |
| ПР3. Итоговая модель | `python src/train.py` | `models/model_v1.joblib`, `models/model_v1.json` |
| ПР4. Демонстрационные порции | `python src/make_bad_batch.py` и `python src/make_stress_batch.py` | `data/demo/bad_batch.csv`, `data/incoming/batch_05.csv` |
| ПР4. Оценка и валидация (≈ 1 мин) | `python src/evaluate.py` | `reports/pr4/` (протокол, метрики, срезы, важность, PSI, паспорт) |
| ПР5. Обработка порции | `python src/process_batch.py data/incoming/batch_01.csv` | записи в `data/credit_system.db`, JSON-сводка |
| ПР5. Панель показателей | `streamlit run app/dashboard.py` |  |
| ПР5. Программный интерфейс | `uvicorn app.api:app --port 8000` |  |
| ПР5. Переобучение / откат | `python src/retrain.py` (`--rollback v1`) | `models/model_v2.joblib`, `current.json` |
| ПР5. Замер времени API | `python src/api_benchmark.py` | `reports/pr5/api_timing.json` |
| ПР5. Диаграммы | `python src/make_diagrams.py` | `diagrams/pr5/` (draw.io + PNG) |
| Тесты | `python -m pytest tests` | 4 теста пайплайна признаков и 5 тестов API |

## Воспроизводимость

- Версии библиотек зафиксированы в `requirements.txt`, начальное значение
  генераторов случайных чисел — `SEED = 42` (`src/config.py`).
- Данные хранятся в Git байт-в-байт (`.gitattributes`); повторный запуск
  `src/split_batches.py` даёт те же файлы. Контрольные суммы MD5:

  | Файл | MD5 |
  |---|---|
  | `data/raw/credit_default.csv` | `128fad338c2f0025f2cd15d6b6d74154` |
  | `data/initial.csv` | `ac1dbdf67284d14cf33fd519269424db` |
  | `data/incoming/batch_01.csv` | `1035ea7ecf3b5fff9024203baf84beb9` |
  | `data/incoming/batch_02.csv` | `c02a558081da1254dd416d50ae549623` |
  | `data/incoming/batch_03.csv` | `9792bfbc5f22aeb5372fbec4c35429e2` |
  | `data/incoming/batch_04.csv` | `a715266e124f89c7b81255d362540495` |
| `data/incoming/batch_05.csv` (создается скриптом) | `30326c7211119c68fc2f90b5ecfd8c12` |
  | `data/train.csv` | `cfef5da3102c7430fced013df0e5180b` |
  | `data/test.csv` | `18f139aaf554afd60781147ce2cbb9ad` |

## Версии по практическим работам

| Тег | Содержание |
|---|---|
| `pr1` | инициализация проекта: структура, зависимости, исходные данные |
| `pr2` | разбиение данных и разведочный анализ (данные под DVC) |
| `pr3` | пайплайн признаков, обучение и сравнение моделей, итоговая модель `model_v1` |
| `pr4` | оценка качества на тесте и порциях, интерпретация, порог 0,25, паспорт модели |
| `pr5` | сервис: хранилище SQLite, сценарий обновления, панель Streamlit, API FastAPI, мониторинг и переобучение (v2) |

После `pr2` проект перестроен по структуре учебно-методического пособия:
данные хранятся непосредственно в Git (объём около 6 МБ, DVC без удалённого
хранилища не позволял получить данные при клонировании), скрипты разбиения и
EDA перенесены в `src/split_batches.py` и `src/eda.py` с сохранением
результатов.
