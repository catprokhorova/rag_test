from pathlib import Path
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
        )
        # Ensure the retriever points to freshly indexed data.
        _retriever = QdrantRetriever()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        status="ok",
        chunks_jsonl=str(chunks_jsonl),
        indexed_chunks=indexed_chunks,
    )
