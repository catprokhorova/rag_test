# rag_test

Локальный чат-бот поддержки Moodle LMS с RAG (выгрузка документации Moodle -> Qdrant -> поиск релевантных фрагментов -> генерация ответов локальной HF-моделью).

## Требования

- Python 3.10+
- Локально доступный runtime для HuggingFace моделей (CPU или GPU)
- Локальный Qdrant

Зависимости:

```bash
pip install -r requirements.txt
```

Примечание: модель(и) для `sentence-transformers` и `transformers` будут скачаны один раз при первом запуске (дальше всё работает локально без внешних API).


## Запуск через Docker Compose (3 контейнера)

В `docker-compose.yml` поднимаются:
- `qdrant` (vector DB)
- `rag` (RAG-сервис: retrieval + локальная LLM, порт `8001`)
- `backend` (фасадный API, порт `8000`, ходит в `rag` по внутренней сети compose)

Запуск:

```bash
docker compose up --build
```

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### Пример чата через backend

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Как создать новый курс в Moodle?","language":"auto"}'
```

## 1) Выгрузка и подготовка документации (offline ingestion)

Скрипт использует MediaWiki API (а не HTML-скрейпинг), чтобы снизить шанс капчи.

Файлы:
- `data/cache/wiki_pages/` (сырой кэш ответов API)
- `data/processed/moodle_chunks.jsonl` (чанки для индексации)

Быстрый запуск (ограничение страниц):

```bash
python -m src.prep.ingest_moodle_docs --max-pages 200 --resume
```

Полный запуск (без `--max-pages`), с возобновлением:

```bash
python -m src.prep.ingest_moodle_docs --resume
```

### Если вместо API приходит CAPTCHA

Если MediaWiki API внезапно возвращает human verification/CAPTCHA, скрипт упадёт с понятной ошибкой (чтобы не пытаться “долбить” бесконечно ретраями).

Вариант обхода по требованиям задания:
1. Один раз решите CAPTCHA в браузере (возможна ручная выгрузка нужных страниц).
2. Скопируйте выгруженные страницы в локальную директорию.
3. Запустите ingestion в режиме `--from-local-dir`.

Ожидаемый формат файлов в `--from-local-dir`:

- директория содержит `*.json` файлы
- каждый файл — JSON-объект вида:
  - `{"title": "...", "wikitext": "...", "html": "...", "url": "optional"}`  
  (минимум `title` + любой из `wikitext/html`)

Команда:

```bash
python -m src.prep.ingest_moodle_docs --from-local-dir /path/to/local_pages --resume
```

## 2) Индексация чанков в Qdrant

```bash
python -m src.backend.scripts.index_qdrant --recreate-collection
```

Если коллекцию пересоздавать не нужно — уберите `--recreate-collection`.

## 3) Запуск REST API

```bash
uvicorn src.backend.api.facade.main:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Чат:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Как создать новый курс в Moodle?", "language":"auto"}'
```

## Быстрая проверка запросов (локально, без REST)

```bash
python -m src.eval.run_eval
```

## Структура проекта

- `src/prep/`: выгрузка Moodle docs, чистка текста, chunking
- `src/backend/infrastructure/`: интеграции и внешние хранилища (Qdrant)
- `src/backend/scripts/`: backend-скрипты (индексация)
- `src/rag/`: embeddings, retriever, generator
- `src/backend/api/`: API-контракты и entrypoints (`facade` и `rag`)
- `src/eval/`: прогон тестовых запросов