from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None, description="Conversation id. If omitted, server creates a new one."
    )
    # 'auto' -> heuristic based on message; or force 'ru'/'en'
    language: Optional[Literal["auto", "ru", "en"]] = Field(
        "auto", description="Answer language."
    )


class SourceItem(BaseModel):
    title: str
    url: str
    chunk_id: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceItem]

