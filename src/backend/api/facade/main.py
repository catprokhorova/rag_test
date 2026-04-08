import os
import uuid

import httpx
from fastapi import FastAPI, HTTPException

from src.backend.api.contracts.schemas import ChatRequest, ChatResponse


app = FastAPI(title="Backend API (facade) for Moodle RAG Bot")

RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://rag:8001").rstrip("/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rag_base_url": RAG_BASE_URL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
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
    data["session_id"] = session_id
    return ChatResponse.model_validate(data)
