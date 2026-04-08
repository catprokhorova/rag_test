import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException

from src.backend.api.contracts.schemas import ChatRequest, ChatResponse
from src.rag.generator import LocalTextGenerator
from src.rag.retriever import QdrantRetriever
from src.config import settings


app = FastAPI(title="Moodle RAG Bot (local, offline)")

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
