from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None, description="Conversation id. If omitted, server creates a new one."
    )
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


class IngestRequest(BaseModel):
    toc_title: str = Field("Table_of_Contents", description="Moodle docs TOC page title.")
    max_pages: Optional[int] = Field(
        None, description="Limit pages for quick ingestion tests."
    )
    resume: bool = Field(
        True, description="Resume using ingest_state.json and append chunks."
    )
    recreate_collection: bool = Field(
        False, description="Drop and recreate Qdrant collection before indexing."
    )


class IngestResponse(BaseModel):
    status: str
    chunks_jsonl: str
    indexed_chunks: int
