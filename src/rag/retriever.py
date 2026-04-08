from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.config import settings
from src.backend.infrastructure.vector_store.qdrant_store import QdrantStore, RetrievedChunk
from src.rag.embeddings import SentenceTransformerEmbedder


@dataclass(frozen=True)
class RetrievedContext:
    chunks: List[RetrievedChunk]

    def format_for_prompt(self, *, max_chars: int = 12000) -> str:
        """
        Build context block for the LLM prompt from retrieved chunks.
        """
        parts: List[str] = []
        used = 0
        for i, c in enumerate(self.chunks, start=1):
            block = f"[{i}] {c.page_title}\n{c.text}"
            if used + len(block) > max_chars and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts).strip()

    def sources(self) -> List[dict]:
        return [
            {
                "title": c.page_title,
                "url": c.url,
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "score": c.score,
            }
            for c in self.chunks
        ]


class QdrantRetriever:
    def __init__(
        self,
        *,
        embedder: Optional[SentenceTransformerEmbedder] = None,
        store: Optional[QdrantStore] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        cfg = settings()
        self.embedder = embedder or SentenceTransformerEmbedder(model_name=cfg.embed_model)
        self.store = store or QdrantStore()
        self.top_k = top_k or cfg.retrieve_top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> RetrievedContext:
        query_vector = self.embedder.embed_query(query)
        chunks = self.store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.score_threshold,
        )
        return RetrievedContext(chunks=chunks)

