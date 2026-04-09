from pathlib import Path
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException

from src.backend.api.contracts.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
)
from src.backend.scripts.index_qdrant import index_chunks
from src.prep.ingest_docs import ingest
from src.rag.generator import LocalTextGenerator
from src.rag.retriever import QdrantRetriever
from src.config import settings


app = FastAPI(title="Docs RAG Bot (local, offline)")
logger = logging.getLogger(__name__)

_sessions: Dict[str, List[Tuple[str, str]]] = {}
_retriever: Optional[QdrantRetriever] = None
_generator: Optional[LocalTextGenerator] = None


@app.on_event("startup")
def _startup() -> None:
    global _retriever, _generator
    _ = settings()
    _retriever = QdrantRetriever()
    _generator = LocalTextGenerator()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    global _retriever, _generator
    if _retriever is None or _generator is None:
        raise HTTPException(status_code=503, detail="Server is not ready yet.")

    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])
    history = history[-4:]

    retrieved = _retriever.retrieve(req.message)
    context = retrieved.format_for_prompt()

    lang = req.language
    if lang == "auto":
        lang = None

    result = _generator.generate(
        question=req.message,
        context=context,
        history=history,
        language=lang,
    )

    answer = result.text.strip()
    _sessions[session_id] = history + [(req.message, answer)]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=[
            {
                "title": source["title"],
                "url": source["url"],
                "chunk_id": source["chunk_id"],
                "chunk_index": source["chunk_index"],
                "score": source["score"],
            }
            for source in retrieved.sources()
        ],
    )


@app.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(req: IngestRequest) -> IngestResponse:
    global _retriever
    cfg = settings()
    chunks_jsonl = Path(cfg.data_dir) / "processed" / "docs_chunks.jsonl"
    started_at = time.perf_counter()
    logger.info(
        "Admin ingest started (max_pages=%s, resume=%s, recreate_collection=%s)",
        req.max_pages,
        req.resume,
        req.recreate_collection,
    )

    try:
        ingest(
            start_urls=req.start_urls,
            allowed_prefixes=req.allowed_prefixes,
            max_pages=req.max_pages,
            output_jsonl=chunks_jsonl,
            resume=req.resume,
        )
        indexed_chunks = index_chunks(
            chunks_jsonl=chunks_jsonl,
            batch_size=cfg.embedding_batch_size,
            recreate_collection=req.recreate_collection,
            embedder=_retriever.embedder if _retriever is not None else None,
        )
        # Avoid reloading the embeddings model after indexing.
        if _retriever is None:
            _retriever = QdrantRetriever()
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.exception("Admin ingest failed after %dms", elapsed_ms)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "Admin ingest completed (indexed_chunks=%d, chunks_jsonl=%s, elapsed_ms=%d)",
        indexed_chunks,
        chunks_jsonl,
        elapsed_ms,
    )

    return IngestResponse(
        status="ok",
        chunks_jsonl=str(chunks_jsonl),
        indexed_chunks=indexed_chunks,
    )
