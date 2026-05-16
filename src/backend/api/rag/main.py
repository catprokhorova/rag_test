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
from src.rag.generator import generate_answer
from src.rag.retriever import QdrantRetriever
from src.config import settings


app = FastAPI(title="Docs RAG Bot (local, offline)")
logger = logging.getLogger(__name__)

_sessions: Dict[str, List[Tuple[str, str]]] = {}
_retriever: Optional[QdrantRetriever] = None


def _run_rag(question: str, history: List[Tuple[str, str]], language: Optional[str]):
    retrieved = _get_retriever().retrieve(question)
    context = retrieved.format_for_prompt()
    if language == "auto":
        language = None
    result = generate_answer(
        question=question,
        context=context,
        history=history,
        language=language,
    )
    return result, retrieved


@app.on_event("startup")
def _startup() -> None:
    global _retriever
    _ = settings()
    # Warm model-dependent services once during startup, so requests only use in-memory instances.
    _retriever = QdrantRetriever()


def _get_retriever() -> QdrantRetriever:
    if _retriever is None:
        # Startup initializes retriever; this is a defensive guard.
        raise RuntimeError("Retriever is not initialized yet.")
    return _retriever


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    global _retriever
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])
    history = history[-4:]

    try:
        result, retrieved = _run_rag(req.message, history, req.language)
    except Exception as exc:
        logger.exception("Retriever initialization/query failed")
        raise HTTPException(status_code=503, detail=f"Retriever unavailable: {exc}") from exc

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
    pdf_dir = Path(req.pdf_dir) if req.pdf_dir else cfg.pdf_dir
    logger.info(
        "Admin ingest started (pdf_dir=%s, max_pdfs=%s, resume=%s, recreate_collection=%s)",
        pdf_dir,
        req.max_pdfs,
        req.resume,
        req.recreate_collection,
    )

    try:
        ingest_outcome = ingest(
            pdf_dir=pdf_dir,
            max_pdfs=req.max_pdfs,
            output_jsonl=chunks_jsonl,
            resume=req.resume,
        )
        indexed_chunks = index_chunks(
            chunks_jsonl=chunks_jsonl,
            batch_size=cfg.embedding_batch_size,
            recreate_collection=req.recreate_collection,
        )
        _retriever = QdrantRetriever()
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.exception("Admin ingest failed after %dms", elapsed_ms)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "Admin ingest completed (indexed_chunks=%d, pdfs_processed=%d, chunks_appended=%d, chunks_jsonl=%s, elapsed_ms=%d)",
        indexed_chunks,
        ingest_outcome.pdfs_processed,
        ingest_outcome.chunks_appended,
        chunks_jsonl,
        elapsed_ms,
    )

    note: Optional[str] = None
    if ingest_outcome.pdfs_processed == 0:
        note = (
            "No PDF files were parsed (directory empty, wrong PDF_DIR, or all files skipped on resume). "
            "indexed_chunks may only reflect re-embedding of existing JSONL. "
            "Mount or copy PDFs into the container and set PDF_DIR."
        )

    return IngestResponse(
        status="ok",
        chunks_jsonl=str(chunks_jsonl),
        indexed_chunks=indexed_chunks,
        ingest_pages_fetched=ingest_outcome.pages_fetched,
        ingest_chunks_appended=ingest_outcome.chunks_appended,
        message=note,
    )
