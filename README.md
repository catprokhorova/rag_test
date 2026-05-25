# rag_test

Local RAG chatbot for LangChain/LangGraph documentation: PDF ingestion → Qdrant → retrieval → **text generation via an OpenAI-compatible HTTP API** (e.g. [LM Studio](https://lmstudio.ai/) on your machine).

## Requirements

- Python 3.10+
- LangChain PDFs on disk (manual download; not crawled from the web)
- A running **chat completions** endpoint compatible with OpenAI’s `/v1/chat/completions` shape (LM Studio’s local server is the typical setup)
- Local **Qdrant** (or a reachable Qdrant instance configured in `.env`)

Install dependencies:

```bash
pip install -r requirements.rag.txt
pip install -r requirements.backend.txt
```

### LLM configuration

Copy `.env.example` to `.env` and set at least:

- **`LLM_CHAT_COMPLETIONS_URL`** — full URL to the chat endpoint, e.g. `http://127.0.0.1:1234/v1/chat/completions` when the RAG process runs on the same host as LM Studio.
- **`LLM_MODEL`** — model id string LM Studio expects for the loaded model (see the LM Studio UI if requests fail with a model error).
- **`PDF_DIR`** — directory containing `*.pdf` files (default: `~/Downloads/langchain`).

Optional: `LLM_API_KEY`, `LLM_REQUEST_TIMEOUT_S`, generation limits (`LLM_MAX_NEW_TOKENS`, etc.). See `.env.example`.

## Run with Docker Compose

Services in `docker-compose.yml`:

- **`qdrant`** — vector database
- **`rag`** — PDF ingest, embeddings, retrieval, and **HTTP calls** to your configured LLM URL (`8001`)
- **`backend`** — facade API (`8000`); forwards chat to `rag`

Place PDFs on the host and point Compose at that folder (default mount: `./pdfs` → `/pdfs` in the container):

```bash
# Option A: symlink or copy PDFs into ./pdfs
mkdir -p pdfs && cp ~/Downloads/langchain/*.pdf pdfs/

# Option B: set host path in .env
# PDF_DIR_HOST=/home/you/Downloads/langchain
```

Start LM Studio (or your server) on the host, then:

```bash
docker compose up --build
```

Inside the `rag` container, `PDF_DIR` is `/pdfs` (set in `docker-compose.yml`). Override the host mount with `PDF_DIR_HOST` in `.env`.

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 1) Ingest PDFs (offline)

PDFs are parsed with LangChain’s `PagedPDFSplitter` (one document per page), then chunked for RAG.

Default input directory: `~/Downloads/langchain` (override with `PDF_DIR` or `--pdf-dir`).

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

Custom PDF directory:

```bash
python -m src.prep.ingest_docs --pdf-dir /path/to/langchain/pdfs
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

With a custom PDF directory inside the container:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_dir":"/pdfs","resume":false,"recreate_collection":true}'
```

## Observability (Langfuse + Phoenix)

The RAG service can send traces to **Langfuse** and **Arize Phoenix** at the same time.

**Langfuse** — set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_BASE_URL` in `.env`.

**Phoenix (local)** — with Phoenix running (e.g. `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`):

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export PHOENIX_PROJECT_NAME=docs-rag
```

Open the UI at [http://localhost:6006](http://localhost:6006). Spans use OpenInference kinds: `RETRIEVER` (Qdrant), `CHAIN` (answer generation), `LLM` (HTTP chat completions). `PHOENIX_AUTO_INSTRUMENT=true` (default) also traces LangChain when `openinference-instrumentation-langchain` is installed.

From Docker, point at the host collector: `PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006`.

See `.env.example` for all observability variables.

## Quick local eval

```bash
python -m src.eval.run_eval
```

## Project structure

- `src/prep/`: PDF ingestion and chunking
- `src/backend/infrastructure/`: integrations and storage (Qdrant)
- `src/backend/scripts/`: indexing scripts
- `src/rag/`: embeddings, retriever, generator (HTTP chat completions)
- `src/backend/api/`: API contracts and entrypoints (`facade` and `rag`)
- `src/eval/`: sample query evaluation
