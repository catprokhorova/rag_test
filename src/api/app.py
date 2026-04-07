import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException

from src.api.schemas import ChatRequest, ChatResponse
from src.rag.generator import LocalTextGenerator
from src.rag.retriever import QdrantRetriever
from src.config import settings


app = FastAPI(title="Moodle RAG Bot (local, offline)")

# In-memory session history. For evaluation tasks this is enough; for production
# you'd use a persistent store.
_sessions: Dict[str, List[Tuple[str, str]]] = {}

_retriever: Optional[QdrantRetriever] = None
_generator: Optional[LocalTextGenerator] = None


@app.on_event("startup")
def _startup() -> None:
    global _retriever, _generator
    cfg = settings()
    # Models are heavy: load them once on startup.
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

    # Keep last N turns to control prompt size.
    max_turns = 4
    history = history[-max_turns:]

    retrieved = _retriever.retrieve(req.message)
    context = retrieved.format_for_prompt()

    lang = req.language
    if lang == "auto":
        lang = None  # generator uses its own heuristic

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
                "title": s["title"],
                "url": s["url"],
                "chunk_id": s["chunk_id"],
                "chunk_index": s["chunk_index"],
                "score": s["score"],
            }
            for s in retrieved.sources()
        ],
    )

