# rag_test

Local RAG chatbot for LangChain/LangGraph documentation: PDF ingestion → Qdrant → retrieval → **text generation via [Yandex Cloud AI Studio](https://yandex.cloud/en/services/ai-studio)** (OpenAI-compatible Responses API).

## Requirements

- Python 3.10+
- LangChain PDFs under `data/` (manual download; not crawled from the web)
- A **Yandex Cloud** folder + API key with access to AI Studio models
- Local **Qdrant** (or a reachable Qdrant instance configured in `.env`)

Install dependencies:

```bash
pip install -r requirements.rag.txt
pip install -r requirements.backend.txt
```

### LLM configuration

Copy `.env.example` to `.env` and set at least:

- **`YANDEX_CLOUD_API_KEY`** — Yandex Cloud API key.
- **`YANDEX_CLOUD_FOLDER`** — cloud folder id (used as `project` and in the `gpt://…` model URI).
- **`YANDEX_CLOUD_MODEL`** — model id, e.g. `aliceai-llm-flash/latest`.

Optional: `DATA_DIR` (default `data`), `YANDEX_CLOUD_BASE_URL`, `LLM_REQUEST_TIMEOUT_S`, generation limits (`LLM_MAX_NEW_TOKENS`, etc.). See `.env.example`.

## Run with Docker Compose

Services in `docker-compose.yml`:

- **`qdrant`** — vector database
- **`rag`** — PDF ingest, embeddings, retrieval, and Yandex Cloud LLM calls (`8001`)
- **`backend`** — facade API (`8000`); forwards chat to `rag`

Put PDFs in `./data` (mounted at `/data` in the `rag` container):

```bash
# Example: copy LangChain PDFs into the project data folder
mkdir -p data && cp ~/Downloads/langchain/*.pdf data/
```

Ensure `YANDEX_CLOUD_API_KEY` is set in `.env`, then:

```bash
docker compose up --build
```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 1) Ingest PDFs (offline)

PDFs are read from `data/*.pdf`, parsed with LangChain’s `PagedPDFSplitter` (one document per page), then chunked for RAG.

Override the input directory with `PDF_DIR` or `--pdf-dir` only if needed.

Output files:

- `data/processed/docs_chunks.jsonl` (chunk payloads)
- `data/processed/ingest_state.json` (resume state)

Quick run:

```bash
python -m src.prep.ingest_docs --resume
```

Limit files for a smoke test:

```bash
python -m src.prep.ingest_docs --max-pdfs 5
```

## 2) Index chunks into Qdrant

```bash
python -m src.backend.scripts.index_qdrant --recreate-collection
```

## 3) Run REST API

```bash
uvicorn src.backend.api.facade.main:app --host 0.0.0.0 --port 8000
```

Chat request example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is LangChain?","language":"auto"}'
```

Admin ingestion via API (parse PDFs + embed into Qdrant):

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"max_pdfs":50,"resume":true,"recreate_collection":true}'
```

## Observability (Langfuse + Phoenix)

The RAG service can send traces to **Langfuse** and **Arize Phoenix** at the same time.

**Langfuse** — set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_BASE_URL` in `.env`.

**Phoenix (local)** — with Phoenix running (e.g. `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`):

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export PHOENIX_PROJECT_NAME=docs-rag
```

Open the UI at [http://localhost:6006](http://localhost:6006). Spans use OpenInference kinds: `RETRIEVER` (Qdrant), `CHAIN` (answer generation), `LLM` (Yandex Responses API). `PHOENIX_AUTO_INSTRUMENT=true` (default) also traces LangChain when `openinference-instrumentation-langchain` is installed.

From Docker, point at the host collector: `PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006`.

See `.env.example` for all observability variables.

## Quick local eval

```bash
python -m src.eval.run_eval
```

## Project structure

- `data/`: source PDFs (`*.pdf`) and `processed/` chunk JSONL
- `src/prep/`: PDF ingestion and chunking
- `src/backend/infrastructure/`: integrations and storage (Qdrant)
- `src/backend/scripts/`: indexing scripts
- `src/rag/`: embeddings, retriever, generator (Yandex Cloud Responses API)
- `src/backend/api/`: API contracts and entrypoints (`facade` and `rag`)
- `src/eval/`: sample query evaluation
