import logging
import os
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from src.backend.api.contracts.schemas import (
    ChatRequest,
    ChatResponseWithSources,
    IngestRequest,
    IngestResponse,
)

app = FastAPI(title="Backend API (facade) for Docs RAG Bot")
logger = logging.getLogger(__name__)

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://rag:8001").rstrip("/")

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rag_base_url": RAG_BASE_URL}

@app.post("/chat", response_model=ChatResponseWithSources)
def chat(req: ChatRequest) -> ChatResponseWithSources:
    session_id = req.session_id or str(uuid.uuid4())
    payload = {
        "session_id": session_id,
        "message": req.message,
        "language": req.language,
    }

    try:
        with httpx.Client(timeout=120) as client:
            response = client.post(f"{RAG_BASE_URL}/chat", json=payload)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"RAG service unavailable: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"RAG error ({response.status_code}): {response.text}"
        )

    data = response.json()
    chat_response = ChatResponseWithSources.model_validate(
        {
            "session_id": data.get("session_id") or session_id,
            "answer": data["answer"],
            "sources": data.get("sources") or [],
        }
    )
    logger.info(
        "chat retrieval session_id=%s chunk_ids=%s",
        chat_response.session_id,
        [source.chunk_id for source in chat_response.sources],
    )
    return chat_response

@app.post("/admin/ingest", response_model=IngestResponse)
def admin_ingest(req: IngestRequest) -> IngestResponse:
    try:
        with httpx.Client(timeout=3600) as client:
            response = client.post(f"{RAG_BASE_URL}/admin/ingest", json=req.model_dump())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"RAG service unavailable: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"RAG error ({response.status_code}): {response.text}"
        )

    return IngestResponse.model_validate(response.json())
