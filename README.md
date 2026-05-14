# rag_test

Local RAG chatbot for LangChain/LangGraph documentation: docs ingestion → Qdrant → retrieval → **text generation via an OpenAI-compatible HTTP API** (e.g. [LM Studio](https://lmstudio.ai/) on your machine).

## Requirements

- Python 3.10+
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

Optional: `LLM_API_KEY`, `LLM_REQUEST_TIMEOUT_S`, generation limits (`LLM_MAX_NEW_TOKENS`, etc.). See `.env.example`.

## Run with Docker Compose

Services in `docker-compose.yml`:

- **`qdrant`** — vector database
- **`rag`** — embeddings, retrieval, and **HTTP calls** to your configured LLM URL (`8001`)
- **`backend`** — facade API (`8000`); forwards chat to `rag`

Start LM Studio (or your server) on the host, then:

```bash
docker compose up --build
```

Health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## 1) Ingest docs (offline)

Default ingestion crawls these seeds:

- `https://docs.langchain.com/oss/python/langchain/overview`
- `https://docs.langchain.com/oss/python/langgraph/overview`

Output files:

- `data/processed/docs_chunks.jsonl` (chunk payloads)
- `data/processed/ingest_state.json` (resume state)

Quick run:

```bash
python -m src.prep.ingest_docs --max-pages 200 --resume
```

Custom crawl boundaries:

```bash
python -m src.prep.ingest_docs \
  --start-url "https://docs.langchain.com/oss/python/langchain/overview" \
  --start-url "https://docs.langchain.com/oss/python/langgraph/overview" \
  --allowed-prefix "https://docs.langchain.com/oss/python/langchain/" \
  --allowed-prefix "https://docs.langchain.com/oss/python/langgraph/" \
  --resume
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

Admin ingestion via API:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d '{"max_pages":50,"resume":true,"recreate_collection":true}'
```

## Quick local eval

```bash
python -m src.eval.run_eval
```

## Project structure

- `src/prep/`: docs ingestion and chunking
- `src/backend/infrastructure/`: integrations and storage (Qdrant)
- `src/backend/scripts/`: indexing scripts
- `src/rag/`: embeddings, retriever, generator (HTTP chat completions)
- `src/backend/api/`: API contracts and entrypoints (`facade` and `rag`)
- `src/eval/`: sample query evaluation
