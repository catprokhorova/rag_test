from contextlib import nullcontext
from pathlib import Path
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import openai
from fastapi import FastAPI, HTTPException
from langfuse import get_client, observe, propagate_attributes

from src.backend.api.contracts.schemas import (
    ChatRequest,
    ChatResponseWithSources,
    IngestRequest,
    IngestResponse,
)
from src.backend.scripts.index_qdrant import index_chunks
from src.config import settings
from src.observability.langfuse_client import langfuse_enabled, shutdown_tracing
from src.observability.phoenix_client import (
    phoenix_enabled,
    setup_phoenix,
    shutdown_phoenix,
)
from phoenix.otel import using_session
from src.prep.ingest_docs import ingest
from src.rag.generator import generate_answer
from src.rag.retriever import QdrantRetriever


app = FastAPI(title="Docs RAG Bot (local, offline)")
logger = logging.getLogger(__name__)

_sessions: Dict[str, List[Tuple[str, str]]] = {}
_retriever: Optional[QdrantRetriever] = None


@app.on_event("startup")
def _startup() -> None:
    global _retriever
    _ = settings()
    if langfuse_enabled():
        get_client()
    setup_phoenix()
    # Warm model-dependent services once during startup, so requests only use in-memory instances.
    _retriever = QdrantRetriever()


@app.on_event("shutdown")
def _shutdown() -> None:
    shutdown_tracing()
    shutdown_phoenix()


def _get_retriever() -> QdrantRetriever:
    if _retriever is None:
        # Startup initializes retriever; this is a defensive guard.
        raise RuntimeError("Retriever is not initialized yet.")
    return _retriever


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _run_chat(req: ChatRequest, session_id: str) -> ChatResponseWithSources:
    history = _sessions.get(session_id, [])
    history = history[-4:]

    try:
        retrieved = _get_retriever().retrieve(req.message)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail=f"Retriever unavailable: {exc}") from exc

    context = retrieved.format_for_prompt()
    lang = None if req.language == "auto" else req.language
    try:
        result = generate_answer(
            question=req.message,
            context=context,
            history=history,
            language=lang,
        )
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        cfg = settings()
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=503,
            detail=(
                f"LLM unreachable at {cfg.yandex_cloud_base_url!r} "
                f"(model=gpt://{cfg.yandex_cloud_folder}/{cfg.yandex_cloud_model}): {exc}"
            ),
        ) from exc
    except openai.APIError as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {exc}") from exc

    sources = retrieved.sources()
    answer = result.text.strip()
    _sessions[session_id] = history + [(req.message, answer)]

    logger.info(
        "chat completed session_id=%s chunk_ids=%s",
        session_id,
        [source["chunk_id"] for source in sources],
    )

    return ChatResponseWithSources(
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
            for source in sources
        ],
    )


@observe(name="docs-rag-chat", capture_input=False, capture_output=False)
def _run_chat_traced(req: ChatRequest, session_id: str) -> ChatResponseWithSources:
    langfuse = get_client()
    langfuse.update_current_span(
        input={"question": req.message, "language": req.language},
    )
    with propagate_attributes(session_id=session_id, tags=["docs-rag"]):
        response = _run_chat(req, session_id)
    langfuse.update_current_span(output={"answer": response.answer})
    return response


def _with_phoenix_session(session_id: str, fn):
    if not phoenix_enabled():
        return fn()
    with using_session(session_id):
        return fn()


@app.post("/chat", response_model=ChatResponseWithSources)
def chat(req: ChatRequest) -> ChatResponseWithSources:
    session_id = req.session_id or str(uuid.uuid4())
    if langfuse_enabled():
        return _with_phoenix_session(
            session_id,
            lambda: _run_chat_traced(req, session_id),
        )
    return _with_phoenix_session(session_id, lambda: _run_chat(req, session_id))


@observe(name="docs-rag-ingest", capture_input=False, capture_output=False)
def _run_ingest(req: IngestRequest) -> IngestResponse:
    global _retriever
    if langfuse_enabled():
        get_client().update_current_span(
            input={
                "pdf_dir": req.pdf_dir,
                "max_pdfs": req.max_pdfs,
                "resume": req.resume,
                "recreate_collection": req.recreate_collection,
            },
        )
        propagate_ctx = propagate_attributes(tags=["docs-rag", "ingest"])
    else:
        propagate_ctx = nullcontext()

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
        with propagate_ctx:
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
            "No PDF files were parsed (data/ empty, wrong pdf_dir, or all files skipped on resume). "
            "indexed_chunks may only reflect re-embedding of existing JSONL. "
            "Place *.pdf files under data/ (mounted at /data in Docker)."
        )

    response = IngestResponse(
        status="ok",
        chunks_jsonl=str(chunks_jsonl),
        indexed_chunks=indexed_chunks,
        ingest_pages_fetched=ingest_outcome.pages_fetched,
        ingest_chunks_appended=ingest_outcome.chunks_appended,
        message=note,
    )
    if langfuse_enabled():
        get_client().update_current_span(
            output={
                "indexed_chunks": response.indexed_chunks,
                "ingest_chunks_appended": response.ingest_chunks_appended,
            }
        )
    return response


@app.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(req: IngestRequest) -> IngestResponse:
    return _run_ingest(req)
