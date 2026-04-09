# rag_test

Local RAG chatbot for LangChain/LangGraph documentation (docs ingestion -> Qdrant -> retrieval -> local generation).

## Requirements

- Python 3.10+
- Local LLM runtime (CPU or GPU)
- Local Qdrant

Install dependencies:

```bash
pip install -r requirements.rag.txt
pip install -r requirements.backend.txt
```

## Run with Docker Compose

Services in `docker-compose.yml`:
- `qdrant` (vector DB)
- `rag` (retrieval + local LLM, `8001`)
- `backend` (facade API, `8000`, calls `rag`)

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
- `src/rag/`: embeddings, retriever, generator
- `src/backend/api/`: API contracts and entrypoints (`facade` and `rag`)
- `src/eval/`: sample query evaluation
