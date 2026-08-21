from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.config import settings


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
    source_file: str = ""
    page: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class ChatResponseWithSources(ChatResponse):
    """RAG-internal response: retrieval metadata for monitoring and attribution."""

    sources: List[SourceItem]


class IngestRequest(BaseModel):
    pdf_dir: Optional[str] = Field(
        None,
        description="Directory with LangChain PDF files (defaults to DATA_DIR / data/).",
    )
    max_pdfs: Optional[int] = Field(
        None, description="Limit PDF files for quick ingestion tests."
    )
    resume: bool = Field(
        False,
        description="Resume using ingest_state.json and append chunks. False rewrites JSONL from scratch.",
    )
    recreate_collection: bool = Field(
        False, description="Drop and recreate Qdrant collection before indexing."
    )


class IngestResponse(BaseModel):
    status: str
    chunks_jsonl: str
    indexed_chunks: int
    ingest_pages_fetched: int = Field(
        0,
        description="PDF files parsed this run (excluding resume skips).",
    )
    ingest_chunks_appended: int = Field(
        0,
        description="Chunk records appended to JSONL this run.",
    )
    message: Optional[str] = Field(
        None,
        description="Optional warning, e.g. when no pages were fetched but indexing still ran.",
    )
